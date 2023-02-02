# Prefrontal-striatal-miniscope

Code for analysis of miniscope Ca²⁺ imaging data associated with
[doi:10.1101/2021.12.01.469698](https://doi.org/10.1101/2021.12.01.469698),
and later versions.

## Requirements

 - MATLAB (tested on R2018a and R2022a)
 - [My own fork of
   octave-bspline](https://github.com/epiasini/octave-bspline), needed
   to define the B-splines we use for temporal expansion. This should
   be automatically cloned as a submodule of this repo, but you still
   need to add it to the matlab path with something like
   `addpath('octave-bspline')` on the matlab prompt. The reason why my
   own fork is recommended over the upstream implementation is that
   mine is much faster when it's used in the way we use it here.
 - [glmnet in
   MATLAB](http://web.stanford.edu/~hastie/glmnet_matlab/index.html). Unfortunately
   the original, official version of this software does not work with
   modern versions of Windows and Mac OS (it still works fine under
   Linux), so I have added to this repo a submodule that pulls in [an
   unofficial fork](https://github.com/epiasini/glmnet_matlab) that
   offers appropriately recompiled binaries. Like octave-bspline
   above, this should be automatically cloned when you clone this
   repo, and you only have to add it to the matlab path with something
   like `addpath('glmnet_matlab')`.
   
## Data format
The data loading procedures in [session.m](session.m) assume that the
data is structured as follows. The data must be saved as a `mat`-file
containing a variable named `msdata_cleaned`. This should be a
structure with a number of fields, hierarchically organized as in the
schematic below.

```
data file
 |
 |-msdata_cleaned
     |
     |---sigfn
     |---mscam_ts
     |---event_ts
     |    |
     |    |-animal_id
     |    |-trial_num
     |    |-event_ts
     |    |-event_name
     |
     |---SessionData
     |    |
     |    |-TrialStartTimestamp
     |    |-Choice
     |    |-Reward
     |
     |---softmaxResult
          |
          |-choices
          |-Qvalues
          |-QDifferences
          |-RPEs
``` 

If the data from the reinforcement learning model (`softmaxResult`) is
not available, this should be gracefully handled. The `session` object
will be built anyway, but if later one tries to add one of the RL
predictors to the design matrix, a warning will be raised.

Remember that multiple `session` objects can be created from multiple
data files, and they can be combined into a single `experiment`
object. This is illustrated in more detail in
[analysis_example.m](analysis_example.m); however, the absolute
minimum workflow is also illustrated below.

## Getting started

### Installation

Use this command to download the code:
```bash
git clone git@gitlab.com:epiasini/prefrontal-striatal-miniscope.git
cd prefrontal-striatal-miniscope
git submodule update --init --recursive
```

Use this command to pull new updates:
```bash
git pull --recurse-submodules
```

Note also that at the very least you'll want to add the
`octave_bspline`, `glmnet_matlab` and `LV_QLearn_2C_SerialReversal`
folders (as well as the root folder of the repo) to your matlab
path. To do this temporarily within a matlab session, use the
following:

```matlab
addpath('octave-bspline', 'glmnet_matlab', 'LV_QLearn_2C_SerialReversal');
```

### Basic usage

To get a sense of how the code may be used, have a look at
[analysis_example.m](analysis_example.m). The tl;dr is as follows:
```matlab
% load data
s1 = session('datafile_1.mat');
s2 = session('datafile_2.mat');

% specify "active trials" for a given neuron (if you don't want the
% whole session to be fitted for that neuron). Note that you can also
% do this later by accessing the neuron as e.sessions(2).neuron(n)...
s2.neurons(1).set_active_trials(40:120);

% configure design matrix
n_knots = 5;
dm_config = designMatrixConfiguration();
dm_config.add_predictor('center_on', n_knots, 0, 4000);
dm_config.add_predictor('init_start', n_knots, 300, 4000);
dm_config.add_predictor('choice_1', n_knots, 2000, 2000);
dm_config.add_predictor('choice_-1', n_knots, 2000, 2000);
dm_config.add_predictor('Q_chosen', n_knots, 5000, 300);
dm_config.add_predictor('Q_difference', n_knots, 1000, 300);
dm_config.add_predictor('RPE', n_knots, 0, 1000);
dm_config.add_predictor('outcome_1', n_knots, 200, 5000);
dm_config.add_predictor('outcome_-1', n_knots, 200, 5000);

% configure regression model
fit_config = glmFitConfiguration(); 

% create experiment object including data from both sessions
e = experiment([s1, s2], dm_config, fit_config);

% visualize kernels (model fits are performed under the hood)
e.plot_kernels();

% visualize neuron tuning (this involves fitting reduced models
% where we remove one predictor at a time)
e.plot_tuning_summary();
```
Running the example above may take a while (the tuning summary is the
longest to run as it involves fitting several variants of the original
model). After it is run, however, all model fits are automatically
cached, so for instance calling again `e.plot_tuning_summary()` would
be almost instantaneous.

## Saving and loading analyses
Note that, since all fit results are automatically stored inside the
objects instantiated by the code, to save the analyses to disk it is
sufficient to save `experiment` object to disk using a `mat`-file. For
instance, the following will result in the tuning summaries being
plotted without having to re-fit all the models.
```matlab
save('model_analysis.mat', 'e')
clear
load('model_analysis.mat')
e.plot_tuning_summary()
```
This is convenient to avoid having to repeat the model fits every time
we want to work on the data.
