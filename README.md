# Person-in-bed detection

This is an attempt at the [Person-in-bed detection challenge](https://analog-garage.github.io/icassp-2025/)
 of the ICASSP 2025 conference.
 The challenge asks us to detect whether or a not a person is on a bed using readings from
 an accelerometer attached to their mattress.
 The accelerometer gives 3D readings for each axis, sampled at 250 Hz, and using this
 time-series data, we need to detect whether a person was on the bed at each sampling time stamp.
 The goal is to detect as accurately as possible, and have a low latency of transition in the
 output when a person moves into the bed or out of it.

 There are two tasks in the challenge.
 In the first one, a person is either on the bed or out of it for the entire duration
 of the given sample, i.e., they don't transition from "in-bed" to "out-of-bed"
 or vice versa during each sample.
 So the output is just a binary value of whether someone was in-bed or out-of-bed.
In the second task, the sample is much longer and there are transition, which makes
latency a relevant metric.
Since this is a relatively more challenging task, this repository is mainly
aimed at the second task.
Converting the output into one for the first task is straightforward &mdash; simply average
the LLRs over time and make the final decision.

## Dependencies

Create a fresh python (virtual) environment and install the following.

```bash
pip install torch torchaudio torchvision
pip install lightning[extra]
```

## Usage

The various config parameters are in `config.yaml`.
You can edit them directly or pass them through the command line.

### Training a model from scratch

To train a model from scratch, run the following.

```bash
python pib.py --config config.yaml --mode train --train.max_epochs 10 --data.data_path <path to train.json>
```

This creates a checkpoint in your results path (check `config.yaml` to edit the path).
By default, this would be `results/transformer/lightning_logs/version_<no.>/checkpoints/<ckpt_filename>.ckpt`
The version number is determined from other environment variables or just increments if the environment
variables are not set.

### Running a model

We provide two pre-trained checkpoints.
One is a small model with around 3.6k parameters more suitable for
embedded devices, and the second is a larger model
with around 63.8k parameters.
The performance is very similar on both, so perhaps it's not worth it
to use the larger model.
The larger model scores slightly higher on the challenge's metrics though.

While you can load and use the model directly using the `--mode predict` option, it might
be easier to use the `.ipynb` notebooks provided, which do this for you.
Open `test-streaming.ipynb` and run the cells.
Edit the paths so that the appropriate paths are used for the `test.json` or `train.json`
file and the checkpoints.
You can also plot the output to see what it looks like, and compare with the labels
if they are given (if you're running `train.json`).
Comment out or comment in the appropriate lines in the last cell which does plotting.

### Creating a new model

If you want to use a custom model instead of the one provided, 
you can add a model to `model.py`.
Use the decorator `@register_model` on your model to register it into
the list of known models.
Then you'll just be able to pass the name of your model as a command-line
argument `--model.name` or edit it in `config.yaml`.
Also make sure to inherit from `utils.PIBMainModel` so that all lightning
functionalities are available in your model
(i.e., you don't have to worry about checkpointing, EMA, etc.).
You should be able to write this as a regular pytorch `nn.Module` model.
If your model uses paramters not used already, you might also have to add new
command-line arguments in the `get_parser` function of `utils.py`.