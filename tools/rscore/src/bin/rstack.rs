use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use liblinear::util::TrainingInput;
use liblinear::{toggle_liblinear_stdout_output, Builder, LibLinearModel, SolverType};

#[derive(Clone)]
struct Config {
    name: String,
    c: f64,
    solver: String,
    eps: f64,
    lambda_l1: Option<f64>,
    lambda_l2: f64,
    passes: usize,
    folds: usize,
    jobs: usize,
    seeds: Vec<u64>,
    min_auc: f64,
    include_meta: bool,
    submissions_dir: PathBuf,
}

struct Model {
    bias: f32,
    weights: Vec<f32>,
}

struct Dataset {
    names: Vec<String>,
    x_cols: Vec<Vec<f32>>,
    x_test_cols: Vec<Vec<f32>>,
    y: Vec<u8>,
    test_ids: Vec<u64>,
}

fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn logit(p: f64) -> f32 {
    let q = p.clamp(1e-7, 1.0 - 1e-7);
    (q / (1.0 - q)).ln() as f32
}

fn soft_threshold(x: f64, t: f64) -> f64 {
    if x > t {
        x - t
    } else if x < -t {
        x + t
    } else {
        0.0
    }
}

fn split_csv_line(line: &str) -> Vec<&str> {
    line.trim_end_matches(['\n', '\r']).split(',').collect()
}

fn find_col(header: &[&str], name: &str) -> Result<usize, Box<dyn Error>> {
    header
        .iter()
        .position(|h| h.trim() == name)
        .ok_or_else(|| format!("missing column {name}").into())
}

fn read_train_y(path: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();
    let header_line = lines.next().ok_or("empty train.csv")??;
    let header = split_csv_line(&header_line);
    let id_col = find_col(&header, "id")?;
    let y_col = find_col(&header, "PitNextLap")?;
    let mut rows = Vec::new();
    for (line_no, line) in lines.enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = split_csv_line(&line);
        let id: u64 = fields
            .get(id_col)
            .ok_or("missing id")?
            .parse()
            .map_err(|e| format!("train.csv:{} bad id: {e}", line_no + 2))?;
        let y_raw: f64 = fields
            .get(y_col)
            .ok_or("missing target")?
            .parse()
            .map_err(|e| format!("train.csv:{} bad target: {e}", line_no + 2))?;
        rows.push((id, if y_raw >= 0.5 { 1 } else { 0 }));
    }
    rows.sort_by_key(|r| r.0);
    Ok(rows.into_iter().map(|(_, y)| y).collect())
}

fn read_test_ids(path: &Path) -> Result<Vec<u64>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();
    let header_line = lines.next().ok_or("empty test.csv")??;
    let header = split_csv_line(&header_line);
    let id_col = find_col(&header, "id")?;
    let mut ids = Vec::new();
    for (line_no, line) in lines.enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = split_csv_line(&line);
        let id: u64 = fields
            .get(id_col)
            .ok_or("missing id")?
            .parse()
            .map_err(|e| format!("test.csv:{} bad id: {e}", line_no + 2))?;
        ids.push(id);
    }
    ids.sort_unstable();
    Ok(ids)
}

fn parse_shape(header: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    let start = header.find('(').ok_or("npy header missing shape start")?;
    let end = header[start..]
        .find(')')
        .ok_or("npy header missing shape end")?
        + start;
    let inside = &header[start + 1..end];
    let mut shape = Vec::new();
    for part in inside.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        shape.push(p.parse::<usize>()?);
    }
    Ok(shape)
}

fn read_npy_1d(path: &Path) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)?;
    if &magic != b"\x93NUMPY" {
        return Err(format!("{} is not an npy file", path.display()).into());
    }
    let mut version = [0u8; 2];
    file.read_exact(&mut version)?;
    let header_len = match version {
        [1, 0] | [2, 0] => {
            let mut len_bytes = [0u8; 2];
            file.read_exact(&mut len_bytes)?;
            u16::from_le_bytes(len_bytes) as usize
        }
        [3, 0] => {
            let mut len_bytes = [0u8; 4];
            file.read_exact(&mut len_bytes)?;
            u32::from_le_bytes(len_bytes) as usize
        }
        _ => return Err(format!("unsupported npy version {:?}", version).into()),
    };
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header = String::from_utf8_lossy(&header_bytes);
    if !header.contains("False") {
        return Err(format!("{}: only C-order arrays supported", path.display()).into());
    }
    let is_f8 = header.contains("'descr': '<f8'") || header.contains("\"descr\": \"<f8\"");
    let is_f4 = header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\"");
    if !is_f8 && !is_f4 {
        return Err(format!("{}: only little-endian f4/f8 arrays supported", path.display()).into());
    }
    let shape = parse_shape(&header)?;
    if shape.len() != 1 {
        return Err(format!("{}: expected 1-D array, got {:?}", path.display(), shape).into());
    }
    let n = shape[0];
    let mut data = Vec::with_capacity(n);
    if is_f8 {
        let mut bytes = vec![0u8; n * 8];
        file.read_exact(&mut bytes)?;
        for chunk in bytes.chunks_exact(8) {
            let mut b = [0u8; 8];
            b.copy_from_slice(chunk);
            data.push(f64::from_le_bytes(b));
        }
    } else {
        let mut bytes = vec![0u8; n * 4];
        file.read_exact(&mut bytes)?;
        for chunk in bytes.chunks_exact(4) {
            let mut b = [0u8; 4];
            b.copy_from_slice(chunk);
            data.push(f32::from_le_bytes(b) as f64);
        }
    }
    Ok(data)
}

fn auc(y: &[u8], pred: &[f32]) -> f64 {
    let mut pairs: Vec<(f32, u8)> = pred.iter().copied().zip(y.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let pos = y.iter().filter(|&&v| v == 1).count();
    let neg = y.len() - pos;
    if pos == 0 || neg == 0 {
        return f64::NAN;
    }
    let mut sum_pos_ranks = 0.0;
    let mut i = 0;
    while i < pairs.len() {
        let mut j = i + 1;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        let avg_rank = ((i + 1) as f64 + j as f64) / 2.0;
        let pos_in_tie = pairs[i..j].iter().filter(|(_, yy)| *yy == 1).count();
        sum_pos_ranks += avg_rank * pos_in_tie as f64;
        i = j;
    }
    (sum_pos_ranks - (pos * (pos + 1) / 2) as f64) / (pos as f64 * neg as f64)
}

fn load_dataset(cfg: &Config) -> Result<Dataset, Box<dyn Error>> {
    let y = read_train_y(Path::new("data/train.csv"))?;
    let test_ids = read_test_ids(Path::new("data/test.csv"))?;
    let mut entries = Vec::new();
    for entry in fs::read_dir(&cfg.submissions_dir)? {
        let path = entry?.path();
        let Some(file_name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !file_name.ends_with("_oof.npy") {
            continue;
        }
        let name = file_name.trim_end_matches("_oof.npy").to_string();
        if !cfg.include_meta
            && (name.starts_with("best_") || name.starts_with("stack_") || name.starts_with("rstack_"))
        {
            continue;
        }
        let test_path = cfg.submissions_dir.join(format!("{name}_test.npy"));
        if test_path.exists() {
            entries.push((name, path, test_path));
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut names = Vec::new();
    let mut x_cols = Vec::new();
    let mut x_test_cols = Vec::new();

    for (name, oof_path, test_path) in entries {
        let oof_raw = read_npy_1d(&oof_path)?;
        let test_raw = read_npy_1d(&test_path)?;
        if oof_raw.len() != y.len() || test_raw.len() != test_ids.len() {
            continue;
        }
        if !oof_raw.iter().all(|v| v.is_finite()) || !test_raw.iter().all(|v| v.is_finite()) {
            continue;
        }
        let oof_pred: Vec<f32> = oof_raw.iter().map(|&v| v as f32).collect();
        let model_auc = auc(&y, &oof_pred);
        if model_auc <= cfg.min_auc {
            continue;
        }
        println!("load {:24} auc={:.9}", name, model_auc);
        names.push(name);
        x_cols.push(oof_raw.iter().map(|&v| logit(v)).collect());
        x_test_cols.push(test_raw.iter().map(|&v| logit(v)).collect());
    }

    Ok(Dataset {
        names,
        x_cols,
        x_test_cols,
        y,
        test_ids,
    })
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn shuffle<T>(v: &mut [T], seed: u64) {
    let mut state = seed.max(1);
    for i in (1..v.len()).rev() {
        let j = (xorshift64(&mut state) as usize) % (i + 1);
        v.swap(i, j);
    }
}

fn stratified_folds(y: &[u8], folds: usize, seed: u64) -> Vec<usize> {
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    for (i, &yy) in y.iter().enumerate() {
        if yy == 1 {
            pos.push(i);
        } else {
            neg.push(i);
        }
    }
    shuffle(&mut pos, seed ^ 0x9e3779b97f4a7c15);
    shuffle(&mut neg, seed ^ 0xbf58476d1ce4e5b9);
    let mut fold_of = vec![0usize; y.len()];
    for (k, idx) in pos.into_iter().enumerate() {
        fold_of[idx] = k % folds;
    }
    for (k, idx) in neg.into_iter().enumerate() {
        fold_of[idx] = k % folds;
    }
    fold_of
}

fn train_cd(
    x_cols: &[Vec<f32>],
    y: &[u8],
    train_idx: &[usize],
    lambda_l1: f64,
    lambda_l2: f64,
    passes: usize,
) -> Model {
    let m = x_cols.len();
    let n = train_idx.len() as f64;
    let pos = train_idx.iter().filter(|&&i| y[i] == 1).count() as f64;
    let rate = (pos + 0.5) / (n + 1.0);
    let mut bias = (rate / (1.0 - rate)).ln() as f32;
    let mut weights = vec![0.0f32; m];
    let mut z = vec![0.0f32; y.len()];
    for &i in train_idx {
        z[i] = bias;
    }

    for pass in 0..passes {
        let mut g = 0.0f64;
        let mut h = 0.0f64;
        for &i in train_idx {
            let p = sigmoid(z[i]) as f64;
            let yy = y[i] as f64;
            g += p - yy;
            h += (p * (1.0 - p)).max(1e-8);
        }
        let delta_b = (-g / h).clamp(-2.0, 2.0) as f32;
        if delta_b.abs() > 1e-8 {
            bias += delta_b;
            for &i in train_idx {
                z[i] += delta_b;
            }
        }

        let mut max_delta = delta_b.abs() as f64;
        for j in 0..m {
            let col = &x_cols[j];
            let old_w = weights[j] as f64;
            let mut gj = lambda_l2 * old_w;
            let mut hj = lambda_l2;
            for &i in train_idx {
                let x = col[i] as f64;
                let p = sigmoid(z[i]) as f64;
                let yy = y[i] as f64;
                gj += x * (p - yy) / n;
                hj += x * x * (p * (1.0 - p)).max(1e-8) / n;
            }
            if hj <= 1e-14 {
                continue;
            }
            let raw = old_w - gj / hj;
            let new_w = soft_threshold(raw, lambda_l1 / hj);
            let delta = (new_w - old_w) as f32;
            if delta.abs() > 1e-7 {
                weights[j] = new_w as f32;
                for &i in train_idx {
                    z[i] += delta * col[i];
                }
                max_delta = max_delta.max(delta.abs() as f64);
            }
        }
        if pass + 1 == passes || pass == 0 || (pass + 1) % 10 == 0 {
            eprintln!("    pass {:3}/{passes} max_delta={:.4e}", pass + 1, max_delta);
        }
        if max_delta < 1e-5 {
            break;
        }
    }
    Model { bias, weights }
}

fn train_liblinear(
    x_cols: &[Vec<f32>],
    y: &[u8],
    train_idx: &[usize],
    c: f64,
    eps: f64,
) -> Result<Model, Box<dyn Error>> {
    let m = x_cols.len();
    let mut labels = Vec::with_capacity(train_idx.len());
    let mut features: Vec<Vec<(u32, f64)>> = Vec::with_capacity(train_idx.len());

    for &i in train_idx {
        labels.push(y[i] as f64);
        let mut row = Vec::with_capacity(m);
        for (j, col) in x_cols.iter().enumerate() {
            row.push(((j + 1) as u32, col[i] as f64));
        }
        features.push(row);
    }

    let input = TrainingInput::from_sparse_features(labels, features)
        .map_err(|e| format!("liblinear input error: {e}"))?;
    let mut builder = Builder::new();
    builder.problem().input_data(input).bias(1.0);
    builder
        .parameters()
        .solver_type(SolverType::L1R_LR)
        .constraints_violation_cost(c)
        .stopping_criterion(eps);
    let model = builder.build_model()?;

    let weights = (0..m)
        .map(|j| model.feature_coefficient((j + 1) as i32, 0) as f32)
        .collect::<Vec<_>>();
    let bias = model.label_bias(0) as f32;
    Ok(Model { bias, weights })
}

fn predict_rows(model: &Model, x_cols: &[Vec<f32>], rows: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len());
    for &i in rows {
        let mut z = model.bias;
        for (w, col) in model.weights.iter().zip(x_cols.iter()) {
            z += *w * col[i];
        }
        out.push(sigmoid(z));
    }
    out
}

fn predict_all(model: &Model, x_cols: &[Vec<f32>], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut z = model.bias;
        for (w, col) in model.weights.iter().zip(x_cols.iter()) {
            z += *w * col[i];
        }
        out[i] = sigmoid(z);
    }
    out
}

fn fold_bag(cfg: &Config, data: &Dataset) -> (Vec<f32>, Vec<f64>) {
    let n = data.y.len();
    let nt = data.test_ids.len();
    let lambda_l1 = cfg
        .lambda_l1
        .unwrap_or_else(|| 1.0 / (cfg.c * (n as f64) * (cfg.folds as f64 - 1.0) / cfg.folds as f64));
    let lambda_l2 = cfg.lambda_l2;
    println!(
        "stack models={} solver={} folds={} jobs={} seeds={:?} passes={} c={} eps={} lambda_l1={:.6e} lambda_l2={:.6e}",
        data.names.len(),
        cfg.solver,
        cfg.folds,
        cfg.jobs,
        cfg.seeds,
        cfg.passes,
        cfg.c,
        cfg.eps,
        lambda_l1,
        lambda_l2,
    );

    let mut oof_sum = vec![0.0f32; n];
    let mut test_sum = vec![0.0f64; nt];
    let mut completed = 0usize;

    for &seed in &cfg.seeds {
        let seed_start = Instant::now();
        let fold_of = stratified_folds(&data.y, cfg.folds, seed);
        let mut seed_oof = vec![0.0f32; n];
        let mut seed_test = vec![0.0f64; nt];
        let mut nonzero = Vec::new();

        let fold_ids = (0..cfg.folds).collect::<Vec<_>>();
        for chunk in fold_ids.chunks(cfg.jobs.max(1)) {
            let results = std::thread::scope(|scope| {
                let mut handles = Vec::new();
                for &fold in chunk {
                    let fold_of_ref = &fold_of;
                    let data_ref = data;
                    let cfg_ref = cfg;
                    handles.push(scope.spawn(move || {
                        let train_idx: Vec<usize> =
                            (0..n).filter(|&i| fold_of_ref[i] != fold).collect();
                        let val_idx: Vec<usize> =
                            (0..n).filter(|&i| fold_of_ref[i] == fold).collect();
                        eprintln!(
                            "  seed={seed} fold={}/{} train={} val={}",
                            fold + 1,
                            cfg_ref.folds,
                            train_idx.len(),
                            val_idx.len()
                        );
                        let mut model = if cfg_ref.solver == "liblinear" {
                            train_liblinear(
                                &data_ref.x_cols,
                                &data_ref.y,
                                &train_idx,
                                cfg_ref.c,
                                cfg_ref.eps,
                            )
                            .expect("liblinear training failed")
                        } else {
                            train_cd(
                                &data_ref.x_cols,
                                &data_ref.y,
                                &train_idx,
                                lambda_l1,
                                lambda_l2,
                                cfg_ref.passes,
                            )
                        };
                        let nz = model.weights.iter().filter(|&&w| w.abs() > 1e-8).count();

                        let mut val_pred = predict_rows(&model, &data_ref.x_cols, &val_idx);
                        let val_y = val_idx.iter().map(|&i| data_ref.y[i]).collect::<Vec<_>>();
                        if auc(&val_y, &val_pred) < 0.5 {
                            model.bias = -model.bias;
                            for w in &mut model.weights {
                                *w = -*w;
                            }
                            val_pred = predict_rows(&model, &data_ref.x_cols, &val_idx);
                        }
                        let test_pred = predict_all(&model, &data_ref.x_test_cols, nt);
                        (fold, val_idx, val_pred, test_pred, nz)
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().expect("fold worker panicked"))
                    .collect::<Vec<_>>()
            });

            for (_fold, val_idx, val_pred, test_pred, nz) in results {
                nonzero.push(nz);
                for (&idx, &p) in val_idx.iter().zip(val_pred.iter()) {
                    seed_oof[idx] = p;
                }
                for (acc, p) in seed_test.iter_mut().zip(test_pred.iter()) {
                    *acc += *p as f64 / cfg.folds as f64;
                }
            }
        }

        completed += 1;
        for (acc, p) in oof_sum.iter_mut().zip(seed_oof.iter()) {
            *acc += *p / cfg.seeds.len() as f32;
        }
        for (acc, p) in test_sum.iter_mut().zip(seed_test.iter()) {
            *acc += *p / cfg.seeds.len() as f64;
        }
        let seed_auc = auc(&data.y, &seed_oof);
        let mut bag_so_far = vec![0.0f32; n];
        let scale = cfg.seeds.len() as f32 / completed as f32;
        for (dst, &v) in bag_so_far.iter_mut().zip(oof_sum.iter()) {
            *dst = v * scale;
        }
        println!(
            "seed={seed:5} seed_auc={:.9} bag_auc={:.9} nz={:.1} time={:.1}s",
            seed_auc,
            auc(&data.y, &bag_so_far),
            nonzero.iter().sum::<usize>() as f64 / nonzero.len() as f64,
            seed_start.elapsed().as_secs_f64(),
        );
    }
    (oof_sum, test_sum)
}

fn write_npy_f64(path: &Path, values: &[f64]) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[1, 0])?;
    let mut header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({},), }}",
        values.len()
    );
    let pad = 16 - ((10 + header.len() + 1) % 16);
    header.extend(std::iter::repeat(' ').take(pad));
    header.push('\n');
    file.write_all(&(header.len() as u16).to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    for &v in values {
        file.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn write_outputs(cfg: &Config, data: &Dataset, oof: &[f32], test: &[f64]) -> Result<(), Box<dyn Error>> {
    let csv_path = cfg.submissions_dir.join(format!("{}.csv", cfg.name));
    let oof_path = cfg.submissions_dir.join(format!("{}_oof.npy", cfg.name));
    let test_path = cfg.submissions_dir.join(format!("{}_test.npy", cfg.name));
    let mut csv = File::create(&csv_path)?;
    writeln!(csv, "id,PitNextLap")?;
    for (&id, &p) in data.test_ids.iter().zip(test.iter()) {
        writeln!(csv, "{id},{p:.12}")?;
    }
    let oof_f64: Vec<f64> = oof.iter().map(|&v| v as f64).collect();
    write_npy_f64(&oof_path, &oof_f64)?;
    write_npy_f64(&test_path, test)?;
    println!("wrote {}", csv_path.display());
    println!("wrote {}", oof_path.display());
    println!("wrote {}", test_path.display());
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let mut cfg = Config {
        name: "rstack_meta_bag".to_string(),
        c: 0.08,
        solver: "liblinear".to_string(),
        eps: 1e-4,
        lambda_l1: None,
        lambda_l2: 0.0,
        passes: 25,
        folds: 5,
        jobs: 2,
        seeds: vec![42, 7, 99, 3407, 1234, 2024],
        min_auc: 0.85,
        include_meta: false,
        submissions_dir: PathBuf::from("submissions"),
    };

    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--name" => {
                i += 1;
                cfg.name = args.get(i).ok_or("--name needs a value")?.clone();
            }
            "--c" => {
                i += 1;
                cfg.c = args.get(i).ok_or("--c needs a value")?.parse()?;
            }
            "--solver" => {
                i += 1;
                cfg.solver = args.get(i).ok_or("--solver needs a value")?.clone();
            }
            "--eps" => {
                i += 1;
                cfg.eps = args.get(i).ok_or("--eps needs a value")?.parse()?;
            }
            "--lambda-l1" => {
                i += 1;
                cfg.lambda_l1 = Some(args.get(i).ok_or("--lambda-l1 needs a value")?.parse()?);
            }
            "--lambda-l2" => {
                i += 1;
                cfg.lambda_l2 = args.get(i).ok_or("--lambda-l2 needs a value")?.parse()?;
            }
            "--passes" => {
                i += 1;
                cfg.passes = args.get(i).ok_or("--passes needs a value")?.parse()?;
            }
            "--folds" => {
                i += 1;
                cfg.folds = args.get(i).ok_or("--folds needs a value")?.parse()?;
            }
            "--jobs" => {
                i += 1;
                cfg.jobs = args.get(i).ok_or("--jobs needs a value")?.parse()?;
            }
            "--seeds" => {
                i += 1;
                cfg.seeds = args
                    .get(i)
                    .ok_or("--seeds needs a value")?
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.trim().parse())
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--min-auc" => {
                i += 1;
                cfg.min_auc = args.get(i).ok_or("--min-auc needs a value")?.parse()?;
            }
            "--include-meta" => cfg.include_meta = true,
            "--submissions-dir" => {
                i += 1;
                cfg.submissions_dir = PathBuf::from(args.get(i).ok_or("--submissions-dir needs a value")?);
            }
            "-h" | "--help" => {
                println!(
                    "usage: rstack [--name NAME] [--c 0.08] [--solver liblinear|cd] [--jobs 2] [--passes 25] \\\n\
                     [--seeds 42,7,99,3407,1234,2024] [--include-meta]\n\n\
                     Default excludes previous best_/stack_/rstack_ meta outputs."
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg {other}").into()),
        }
        i += 1;
    }
    if cfg.folds < 2 || cfg.seeds.is_empty() {
        return Err("need at least two folds and one seed".into());
    }
    if cfg.jobs == 0 {
        return Err("--jobs must be at least 1".into());
    }
    if cfg.solver != "liblinear" && cfg.solver != "cd" {
        return Err("--solver must be liblinear or cd".into());
    }
    Ok(cfg)
}

fn main() -> Result<(), Box<dyn Error>> {
    toggle_liblinear_stdout_output(false);
    let t0 = Instant::now();
    let cfg = parse_args()?;
    let data = load_dataset(&cfg)?;
    println!(
        "loaded {} models, train={}, test={}",
        data.names.len(),
        data.y.len(),
        data.test_ids.len()
    );
    let (oof, test) = fold_bag(&cfg, &data);
    println!("\n{} OOF AUC: {:.9}", cfg.name, auc(&data.y, &oof));
    let mean = test.iter().sum::<f64>() / test.len() as f64;
    let var = test.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / test.len() as f64;
    println!(
        "test stats mean={:.8} std={:.8} min={:.8} max={:.8}",
        mean,
        var.sqrt(),
        test.iter().copied().fold(f64::INFINITY, f64::min),
        test.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    write_outputs(&cfg, &data, &oof, &test)?;
    println!("elapsed {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}
