# Modeling punishment as a rational communicative social action

## Abstract
When deciding whether and how to punish, people consider not only the potential direct consequences, but also, how their choice will affect observers’ judgements about the values and motives underlying the choice. We formalize the decision to punish as a rational communicative social action (RCSA). The model generates rational decisions to punish, incorporating anticipated observers’ judgements obtained from a recursive model of inference using an intuitive theory of mind. Using this model, we synthesize patterns of human punishment from recently published papers. RCSA thus offers a formal model of the cognitive process that humans use to balance preferences for how they are perceived, with other goals for punishing.

## Project structure
The figures in the paper are produced in the following three jupyter notebooks:
- <code>conceptual_figure.ipynb</code>: generate Figure 2
- <code>experimental_data_plots.ipynb</code>: analyze the data from Batistoni et al. (2022), J. J. Jordan and Rand (2020), and Rai (in press) and generate panels A, B, C and D of Figure 3
- <code>model_simulation_plots.ipynb</code>: simulate RCSA model for Batistoni et al. (2022), J. J. Jordan and Rand (2020), and Rai (in press) experiments and generate panels E, F, G and H of Figure 3

All the figures will be automatically saved in the 'results' folder.

### Experimental data
The data for each paper can be found in the folder <code>data/[paper_name]</code>, as well as the code to preprocess the data. 

### Model simulation
The flow of the simulation for each experiment goes as follows:
- The <code>models</code> folder contains the classes that define base punisher, audience and communicative punishers. 
- The 'config' file associated with each experiment contains the model specification for that experiment, e.g., the utility terms and parameters for the base punisher, audience and communicative punisher. All the config files are located in the <code>configs</code> folder.
- The <code>simulate_model.py</code> uses the information in the config file to instantiate the punisher and audience models and run the simulations. The <code>simulate_Jordan_model.py</code> is used for the J. J. Jordan and Rand (2020) experiment.
- The simulated data for each experiment is saved in <code>model_data/paper/[paper_name]</code> folder. This folder contains a preprocessing file as well, that prepares the data suitable for plotting (used in <code>experimental_data_plots.ipynb</code>)

## Reference
```
???
```
