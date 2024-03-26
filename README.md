# A Transformer approach for Electricity Price Forecasting

This is the official implementation of [A Transformer approach for Electricity Price Forecasting](https://arxiv.org/abs/2403.16108).

In this research a Transformer model for Electricity Price Forecasting (EPF) is presented and compared against the state-of-the-art models following the open-source library epftoolbox to enhance reproducibility and transparency in EPF research.


## How to run the experiments

All the code is inside the src folder. To run the training for one model the following can be executed:

```
python -m src.train
```

Inside the main function of the train file, three variables manage the execution:

- dataset: to choose which dataset.
- exec_mode: to choose between train and evaluation of a model.
- save_model: To choose in evaluation if this model is saved as the best model for a specific dataset.

Then, to compare the results of the best models against the results of the models from the epftoolbox, the benchmark file is executed:

```
python -m src.benchmark
```

This benchmark has a variable in its main function called benchmark too, that manages the type of execution:

- benchmark = dnn_last_year: to compute the results of the DNN (state-of-the-art according to epftoolbox) normalizing with the last year's data.
- benchmark = dnn_all_past: to compute the results of the DNN (state-of-the-art according to epftoolbox) normalizing with all the previous data.
- benchmark = naive: to compute the results of a naive model that uses the values of today as the forecast for the following day.
- benchmark = results: compute the final comparison between all the models. To do that, there has to be a best model for each dataset and the three previous modes of execution have to be previously launched. 


## Dependencies

First to install the basic dependencies use the requirements file. Run the following command:

```
pip install -r requirements.txt
```

After this, it will be needed to install the epftoolbox python library that will be used to fetch the different datasets and implement the forecasting comparison. To do that, the following can be executed:

```
git clone https://github.com/jeslago/epftoolbox.git
cd epftoolbox
git checkout 7456ab84b42240b9c2519fb3b1bbbc52868a0817
pip install .
```

## Cite

Please cite our [paper](https://arxiv.org/abs/2403.16108) if you find it useful:


```
@misc{gonzalez2024transformer,
      title={A Transformer approach for Electricity Price Forecasting}, 
      author={Oscar Llorente Gonzalez and Jose Portela},
      year={2024},
      eprint={2403.16108},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```