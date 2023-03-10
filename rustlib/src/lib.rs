use pyo3::prelude::*;
use itertools::Itertools;
use rayon::prelude::*;
use rand::random;

fn ndcg(prediction: &Vec<f64>, y: &Vec<u32>, at: usize) -> f64 {
    let minat = std::cmp::min(at, y.len());
    let iter_sorted_y = y.iter()
        .sorted_by(|a, b| b.partial_cmp(a).unwrap()) // sorting in reverse
        .take(minat);
    let iter_sorted_prediction = prediction.iter()
        .zip(y.iter())
        .map(|(x,y)| (x, random::<u32>(), y))
        .sorted_by(|a,b| b.partial_cmp(a).unwrap()) // sorting in reverse
        .take(minat);
    let (dcg, idcg) = iter_sorted_prediction.zip(iter_sorted_y)
        .enumerate()
        .fold((0.0, 0.0),|(_dcg,_idcg), (i, ((_, _, y_pred), y_ref))| {
            let denominator = (i as f64+2.0).log2();
            let newdcg = _dcg + (2u64.pow(*y_pred) as f64-1.0)/denominator;
            let newidcg = _idcg + (2u64.pow(*y_ref) as f64-1.0)/denominator;
            (newdcg, newidcg)
        });
    if idcg == 0.0 {
        return 0.0;
    }
    dcg/idcg
}

fn transpose<T>(v: &Vec<Vec<T>>) -> Vec<Vec<T>> where T: Clone {
    assert!(!v.is_empty());
    (0..v[0].len())
        .map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<T>>())
        .collect()
}

#[pyclass]
struct WeakRanker {
    docs_per_qid: Vec<Vec<Vec<f64>>>,
    y_per_qid: Vec<Vec<u32>>,
    ndcgs: Vec<Vec<f64>>
}

#[pymethods]
impl WeakRanker {
    #[new]
    fn new(docs_per_qid: Vec<Vec<Vec<f64>>>, y_per_qid: Vec<Vec<u32>>) -> Self {
        let mut ndcgs = vec![vec![]];
        (&docs_per_qid, &y_per_qid).into_par_iter()
            .map(|(xds, yd)| {
                transpose(xds).iter()
                    .map(|feature| {
                        ndcg(&feature, &yd, 10)
                    })
                .collect()
            })
            .collect_into_vec(&mut ndcgs);
        WeakRanker { docs_per_qid, y_per_qid, ndcgs }
    }

    /// Computes alpha for the weak rankers
    fn get_alpha(&self, p_docs: Vec<f64>, h: usize) -> f64 {
        let (num, denom) = (&self.docs_per_qid, &self.y_per_qid, p_docs).into_par_iter()
            .map(|(x_docs_transpose, y_docs, p)| {
                let x_doc = x_docs_transpose.iter().map(|x| x[h]).collect();
                let cur_score = ndcg(&x_doc, &y_docs,10);
                (p * (1.0 + cur_score), p * (1.0 - cur_score))
            })
            .reduce(|| (0.0, 0.0), |a,b| (a.0+b.0, a.1+b.1));
        0.5 * (num/denom).ln()
    }

    fn get_h_alpha(&self, p_docs: Vec<f64>) -> (usize, f64) {
        let mut feature_importance_vec = vec![vec![]];
        self.ndcgs.par_iter()
            .zip(&p_docs)
            .map(|(feature, p)| feature.iter().map(|x| x*p).collect())
            .collect_into_vec(&mut feature_importance_vec);
        let mut feature_importance = vec![0.0; feature_importance_vec[0].len()];
        feature_importance_vec.iter()
            .for_each(|v| 
                v.iter()
                 .enumerate()
                 .for_each(|(i, x)| feature_importance[i] += x)); let mut h = 0; for i in 1..feature_importance.len() { if feature_importance[i] > feature_importance[h] {
                h = i;
            }
        }
        (h, self.get_alpha(p_docs, h))
    }
}

#[pyfunction]
fn ndcg10(prediction: Vec<f64>, y: Vec<u32>) -> PyResult<f64> {
    Ok(ndcg(&prediction, &y, 10))
}

#[test]
fn test_ndcg() {
    let predictions = vec![0.3,0.2,0.5,0.4,0.1,0.6];
    let ys = vec![1,0,0,1,2,2];
    assert!((ndcg(&predictions, &ys, 10) - 0.858475).abs() < 0.00001)
}

#[test]
fn transpose_test() {
    let v = vec![vec![1,2,3],vec![4,5,6],vec![7,8,9]];
    let vt = vec![vec![1,4,7],vec![2,5,8],vec![3,6,9]];
    assert_eq!(transpose(v),vt);
}

/// A Python module implemented in Rust.
#[pymodule]
fn rustlib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ndcg10, m)?)?;
    m.add_class::<WeakRanker>()?;
    Ok(())
}
