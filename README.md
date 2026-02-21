<h1 align=center style="margin:100"> Supervised Contrastive Fine-Tuning for Active Few-Shot Learning </h1>
<p align=center> <i> Accepted at the 2026 International Conference on Language Resources and Evaluation (LREC 2026) </i> </p>

## Abstract
Active Few-Shot Learning (AFSL) is an effective paradigm for improving the performance of large language models under limited annotation budgets. To address the inefficiency of conventional fine-tuning objectives in AFSL, this paper proposes a supervised contrastive fine-tuning framework specifically designed for natural language processing (NLP) text classification tasks. By integrating Supervised Contrastive Learning (SCL) with Hard Negative Mining (HNM), the proposed framework optimizes the embedding space through an enhanced hybrid loss function, thereby improving the utilization efficiency of labeled samples. Extensive experiments on five benchmark datasets show that, under a fixed state-of-the-art (SOTA) query strategy, our method consistently outperforms baseline models in text classification performance. These findings demonstrate that optimizing how to learn—through improved learning objectives—provides a complementary direction to existing query strategies in advancing AFSL.

<hr>

## 📊 Standardized Benchmark Splits (LREC 2026 Resource Contribution)
To ensure rigorous and fair comparisons in future Active Few-Shot Learning (AFSL) research, we eliminate the substantial variance caused by data selection by releasing our exact few-shot splits. 

Please download the `lrec2026_benchmark_splits.zip` file from the **[Releases](https://github.com/zzz-Sgr/AFSL-SCL-HNM/releases)** page of this repository. **Note: Please unzip this file to get the JSON file before running the experiments.** It contains the exact reproducible random splits (for 5 distinct random seeds) used in our experiments across all 5 datasets. Future researchers are highly encouraged to use these standardized indices to draw their initial unlabeled pools and test sets to establish a fair ground for comparison.

## Datasets
Our experiments are conducted on the following datasets. Please download and place them in the `./data/` directory before running the code.
- [MPQA](https://github.com/theSaeed/opinion-mining-using-llms)
- [AG News (Kaggle)](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- [Amazon Reviews (Kaggle)](https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi)

## Usage

1. Create and activate the required Conda environment:

```bash
conda env create -f afl-env.yml
conda activate afl-env

2.Modify the parameters in main.py or the first cell of afl.ipynb as needed. Ensure that your relative paths pointing to the ./data/ folder are correct.

3.Run the experiments using Papermill via main.py:
```bash
python main.py
