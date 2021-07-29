# Introduction
This is a package to supply all the models used by REINVENT.

# Develop
## Setup environment
You can use Conda to create an environment with all the necessary packages installed.

```
$ conda env create -f reinvent_models
[...]
$ conda activate reinvent_models
```

## Run tests
The tests use the `unittest` package testing framework.  Before you can run the tests make sure that you have created a
`config.json`file in the `reinvent_models/configs` directory.  There is an example config in the same directory, which 
you can base your own config off of.  Make sure that you set `MAIN_TEST_PATH` to a non-existent directory; it is where 
temporary files will be written during the tests; if it is set to an existing directory, that directory will be removed 
once the tests have finished.

Once you have created a config file, you can run the tests, located in the 
`testing` directory, by running

```
$ python main_test.py
```
