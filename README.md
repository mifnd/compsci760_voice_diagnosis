# CompSci760 - Voice Diagnosis
Welcome to the GitHub of the voice diagnosis project from 
CompSci760 at University of Auckland. Exciting things are happening.

### Setup:
We use Conda (Anaconda or Miniconda) to manage our python environment. In order to
set up an environment with all the correct packages, cd into the folder with the 
`environments.yml` file and run

`conda env create --name <YourEnvironmentName> --file environment.yml`

You should now be ready to run the project. 

If the `environment.yml` gets updated with new packages, you can use the 
following command to update your conda-environment:

`conda env update --file environment.yml --prune`
