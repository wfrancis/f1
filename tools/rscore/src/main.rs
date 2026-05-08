use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Score {
    path: String,
    auc: f64,
    n: usize,
    pos: usize,
    neg: usize,
    mean_pred: f64,
}

fn split_csv_line(line: &str) -> Vec<&str> {
    line.trim_end_matches(['\n', '\r']).split(',').collect()
}

fn find_col(header: &[&str], names: &[&str]) -> Result<usize, Box<dyn Error>> {
    for name in names {
        if let Some(idx) = header.iter().position(|h| h.trim() == *name) {
            return Ok(idx);
        }
    }
    Err(format!("missing column; looked for one of {:?}", names).into())
}

fn read_labels(path: &Path) -> Result<HashMap<u64, u8>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();
    let header_line = lines.next().ok_or("empty labels file")??;
    let header = split_csv_line(&header_line);
    let id_col = find_col(&header, &["id"])?;
    let y_col = find_col(&header, &["y", "orig_label", "PitNextLap"])?;

    let mut labels = HashMap::new();
    for (line_no, line) in lines.enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = split_csv_line(&line);
        let id: u64 = fields
            .get(id_col)
            .ok_or("missing id field")?
            .parse()
            .map_err(|e| format!("{}:{} bad id: {e}", path.display(), line_no + 2))?;
        let y_raw: f64 = fields
            .get(y_col)
            .ok_or("missing label field")?
            .parse()
            .map_err(|e| format!("{}:{} bad label: {e}", path.display(), line_no + 2))?;
        let y = if y_raw >= 0.5 { 1 } else { 0 };
        labels.insert(id, y);
    }
    Ok(labels)
}

fn auc(labels_and_scores: &mut [(u8, f64)]) -> Result<(f64, usize, usize), Box<dyn Error>> {
    labels_and_scores.sort_by(|a, b| a.1.total_cmp(&b.1));

    let pos = labels_and_scores.iter().filter(|(y, _)| *y == 1).count();
    let neg = labels_and_scores.len().saturating_sub(pos);
    if pos == 0 || neg == 0 {
        return Err("AUC needs at least one positive and one negative label".into());
    }

    let mut sum_pos_ranks = 0.0;
    let mut i = 0;
    while i < labels_and_scores.len() {
        let mut j = i + 1;
        while j < labels_and_scores.len() && labels_and_scores[j].1 == labels_and_scores[i].1 {
            j += 1;
        }
        let avg_rank = ((i + 1) as f64 + j as f64) / 2.0;
        let pos_in_tie = labels_and_scores[i..j]
            .iter()
            .filter(|(y, _)| *y == 1)
            .count();
        sum_pos_ranks += avg_rank * pos_in_tie as f64;
        i = j;
    }

    let auc = (sum_pos_ranks - (pos * (pos + 1) / 2) as f64) / (pos as f64 * neg as f64);
    Ok((auc, pos, neg))
}

fn score_file(path: &Path, labels: &HashMap<u64, u8>) -> Result<Score, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();
    let header_line = lines.next().ok_or("empty prediction file")??;
    let header = split_csv_line(&header_line);
    let id_col = find_col(&header, &["id"])?;
    let pred_col = find_col(&header, &["PitNextLap", "prediction", "pred", "score"])?;

    let mut pairs = Vec::with_capacity(labels.len());
    let mut pred_sum = 0.0;
    for (line_no, line) in lines.enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = split_csv_line(&line);
        let id: u64 = fields
            .get(id_col)
            .ok_or("missing id field")?
            .parse()
            .map_err(|e| format!("{}:{} bad id: {e}", path.display(), line_no + 2))?;
        let Some(&y) = labels.get(&id) else {
            continue;
        };
        let pred: f64 = fields
            .get(pred_col)
            .ok_or("missing prediction field")?
            .parse()
            .map_err(|e| format!("{}:{} bad prediction: {e}", path.display(), line_no + 2))?;
        if !pred.is_finite() {
            return Err(format!("{}:{} non-finite prediction", path.display(), line_no + 2).into());
        }
        pred_sum += pred;
        pairs.push((y, pred));
    }

    let n = pairs.len();
    if n == 0 {
        return Err(format!("{} had no rows matching labels", path.display()).into());
    }
    let (auc, pos, neg) = auc(&mut pairs)?;
    Ok(Score {
        path: path.display().to_string(),
        auc,
        n,
        pos,
        neg,
        mean_pred: pred_sum / n as f64,
    })
}

fn usage(program: &str) {
    eprintln!(
        "usage: {program} [--labels local_labels/test_v4_key.csv] submissions/*.csv\n\
         scores simple id,prediction CSV files with ROC AUC on local pseudo labels"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().collect::<Vec<_>>();
    let program = args.remove(0);
    let mut labels_path = PathBuf::from("local_labels/test_v4_key.csv");
    let mut pred_paths = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--labels" => {
                i += 1;
                if i >= args.len() {
                    usage(&program);
                    return Err("--labels needs a path".into());
                }
                labels_path = PathBuf::from(&args[i]);
            }
            "-h" | "--help" => {
                usage(&program);
                return Ok(());
            }
            arg => pred_paths.push(PathBuf::from(arg)),
        }
        i += 1;
    }

    if pred_paths.is_empty() {
        usage(&program);
        return Err("no prediction files supplied".into());
    }

    let labels = read_labels(&labels_path)?;
    eprintln!("labels: {} rows from {}", labels.len(), labels_path.display());

    let mut scores = Vec::new();
    for path in pred_paths {
        scores.push(score_file(&path, &labels)?);
    }
    scores.sort_by(|a, b| b.auc.total_cmp(&a.auc));

    println!("{:>12} {:>8} {:>8} {:>8} {:>11}  file", "auc", "n", "pos", "neg", "mean_pred");
    for s in scores {
        println!(
            "{:12.9} {:8} {:8} {:8} {:11.6}  {}",
            s.auc, s.n, s.pos, s.neg, s.mean_pred, s.path
        );
    }
    Ok(())
}
