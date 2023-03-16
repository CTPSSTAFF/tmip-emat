from typing import List, Union, Mapping
import yaml
import warnings
import os, sys, time
from shutil import copyfile, copy, rmtree, copytree
import glob
import pandas as pd
import numpy as np

from emat.scope.scope import Scope
from emat.database.database import Database
from emat.model.core_files import FilesCoreModel
from emat.model.core_files.parsers import TableParser, loc, iloc, loc_sum,FileParser
from emat.util.docstrings import copydoc
from emat.exceptions import MissingArchivePathError, ReadOnlyDatabaseError, MissingIdWarning
#from .tdm23_lookups import set_input_datasets, set_other_input_datasets, set_output_datasets, set_input_subfolders, set_metric_datasets

from .process import watchdog,sub_thread #,RepeatTimer
import threading

from .caliperpy import caliper3 as cp

from emat.util.loggers import get_module_logger
_logger = get_module_logger(__name__)

class TDM23_EMAT(FilesCoreModel):
    """
    Setup connections and paths to tdm23

    Args:
        configuration:
            The configuration for this
            core model. This can be passed as a dict, or as a str
            which gives the filename of a YAML file that will be
            loaded.
        scope:
            The exploration scope, as a Scope object or as
            a str which gives the filename of a YAML file that will be
            loaded.
        safe:
            Load the configuration YAML file in 'safe' mode.
            This can be disabled if the configuration requires
            custom Python types or is otherwise not compatible with
            safe mode. Loading configuration files with safe mode
            off is not secure and should not be done with files from
            untrusted sources.
        db:
            An optional Database to store experiments and results.
        name:
            A name for this model, given as an alphanumeric string.
            The name is required by ema_workbench operations.
            If not given, "CTPS" is used.
    """

    
    def __init__(self,
                 configuration:Union[str,Mapping],
                 scope: Union[Scope, str],
                 safe:bool=True,
                 db:Database=None,
                 name:str='tdm23'
                 ):
        super().__init__(
                 configuration=configuration,
                 scope=scope,
                 safe=safe,
                 db=db,
                 name=name,
        )
        self.completedSteps = None
        self.sub = sub_thread(self.completedSteps)
        self.sub.daemon = True
        self.tc = None
        
        self.model_path     = os.path.normpath(self.config['model_path']) 
        self.archive_path   = os.path.normpath(self.config['model_archive'])
        self.parent_scen    = 'AssignTransitOnly' #TODO: set in scope file
        self.scen_name      = 'emat'
        self.scenario       = os.path.join(self.parent_scen, self.scen_name)

        # derived values based on tdm23 structure
        self.tdm_ui         = self.model_path + 'ui/tdm23_ui.dbd'
        self.scenario_path  = os.path.join(self.model_path, "outputs", self.scenario)

        self.scope_name = scope.name
        self.scenario_values = {}
        self.model_obj = None

        # build reverse hash table to lookup post-processing macros given a performance measure
        # k: macro name
        # v[0]: python method to call
        # v[1]: list of performance metrics
        # v[2]: list of TransCAD datasets passed to macro
        '''
        self.__TRANSCAD_MACRO_BY_PM = {}
        for k, v in self.__PM_BY_TRANSCAD_MACRO.items():
            for pm in v[1]:
                self.__TRANSCAD_MACRO_BY_PM[pm] = v[0], k, v[2]
                # print ("pm",pm,self.__TRANSCAD_MACRO_BY_PM[pm])
        '''
        self._parsers = self.__MEASURE_PARSERS                   
        # parser =   self._parsers[0]
        # print ("parser.measure_names", parser.measure_names)

    def setup(self, expvars: dict):
        """
        Configure the core model with experiment variable values

        Args:
            expvars (dict): dictionary of experiment variables

        Raises:
            KeyError: if experiment variable is not supported by the core model
        """
        print("Started model setup at {0}".format(time.strftime("%Y_%m_%d %H%M%S")))

        #start transcad if it isn't already running
        if self.tc is None:
            self.start_transcad()

        #call the pre_process method to refresh input datasets
        #pre_process starts an instance of transcad
        self.pre_process()

        # initialize emat scenario to adapt to specific experiment
        self.scenario_values = {}

        variables_done = []
        for xv in expvars:
            if xv == '_experiment_id_': continue
            if xv in variables_done: continue

            _logger.info(f"\t\t\tSetting experiment variable for {xv} to {expvars[xv]}")
            print("Setting experiment variable for {xv} to {expvars[xv]}")


            try:
                func, macro, args = self.__METHOD_BY_EVAR[xv]
                variables_done += func(self, macro, args, xv, expvars)
                      
            except KeyError:
                _logger.exception("Experiment variable method not available")
                raise
        
        # set values to scenario
        self.__load_model()        
        self.model_obj.CreateScenario(self.parent_scen, self.scen_name, "emat scenario")           
        self.model_obj.ClearScenarioValues(self.scenario)
        self.model_obj.SetScenario(self.scenario)
        self.model_obj.SetScenarioValues(self.scenario_values)

        print("Completed model setup at {0}\n".format(time.strftime("%Y_%m_%d %H%M%S")))


    def pre_process(self):
        """
        TODO: define necessary work - perhaps delete scenario output folder
        """


    def run(self):
        """
        Run the model.

        TransCAD should already have been started and inputs set with the 'pre_process' method.
        
        """

        #start transcad if it isn't already running
        if self.tc is None:
            self.start_transcad()

        # load model
        self.__load_model(scen = self.scenario)

        #delete any existing model reports and logs from the output folder
        #for file in glob.glob(os.path.join(self.model_path, "scenarios", self.scenario, "Out", "massdot*.xml")):
        #    os.remove(file)
            
        print("Starting model run at {0}".format(time.strftime("%Y_%m_%d %H%M%S")))
        _logger.info("Starting model run at {0}".format(time.strftime("%Y_%m_%d %H%M%S")))
        

        if self.sub.is_alive():
            pass
        else:
            self.sub.start()
        # sub.run()
        # kill daemon threads  as soon as the main program exits
        
        try:
            print('{:<20}|{:<10}|{:<10}|{:<10}|{:<10} | {:<20}    | {:<20}'.format(
                 "Parent","pid","status","mem%","cpu%","timestamp","step_done"))
            print("="*120)  
            status = self.model_obj.RunModel({})
            
        except Exception as e:
            print ("!"*10 + str(e))
            print("Main thread is interrupted")
            # thread sub can be killed by using sub.join()
            self.sub.raise_exception()
            # sub.join(timeout=1)
            
            # print (sub.is_alive())
            
            # raise KeyboardInterrupt()
            
            print (self.sub.is_alive())

            sys.exit()
        
        self.completedSteps = self.model_obj.GetCompletedSteps()
        self.sub.completedSteps  = self.completedSteps
        #print(self.tc.ShowArray(["completed steps:",completedSteps]))
        
        print("Completed model run at {0}\n".format(time.strftime("%Y_%m_%d %H%M%S")))
        _logger.info("Completed model run at {0}".format(time.strftime("%Y_%m_%d %H%M%S")))


    def post_process(self,
                     params: dict,
                     measure_names: List[str],
                     output_path=None):
        """
        Runs post processors associated with measures

        For the CTPS model, this method calls TransCAD macros to generate output files

        The model should have previously been executed using the 'run' method

        Args:
            params (dict):
                Dictionary of experiment variables - indices are
                variable names, values are the experiment settings
            measure_names (List[str]):
                List of measures to be processed
            output path (str):
                Path to model output folder (Scenario-dependent in CTPS model)

            Raises:
                KeyError:
                    If post process macro is not available for the specified measure
        """

        pm_done = []
        for pm in measure_names:
            # skip if performance measure handled by other macro
            print ("measure_names: %s"%pm)
            if pm in pm_done: continue
            
            try:
                # func, macro = self.__TRANSCAD_MACRO_BY_PM[pm]
                pm_done += [1.0]
            except KeyError:
                _logger.exception(f"Post process method for pm {pm} not available")
                raise

        results = {}
        for outcome in self.outcomes:
            for entry in outcome.variable_name:
                try:
                    output = float(1) # types other than float is dangerous for 
                except :
                    _logger.warning(self.com_warning_msg.format(entry))
                    raise
                else:
                    results[entry] = output

        return results

    def get_experiment_archive_path(self, experiment_id: int):
        """
        returns the full path to the archive folder for the specified experiment
        """
        if self.archive_path is None:
           # raise MissingArchivePathError('no archive path has been set')
           raise ('no archive path has been set')
        mod_results_path = os.path.join(
            self.archive_path,
            "scp_" + self.scope_name,
            "exp_" + str(experiment_id).zfill(3)
            )
        return mod_results_path
            
    def archive(self, params: dict, archive_folder: str, experiment_id: int=0):
        """
        Copies experiment variables, selected TransCAD input and output datasets, and TransCAD reports
        from the most recent model run to the model archive.

        Args:
            params (dict): dictionary of experiment variables
            archive_folder: full path to the experiment archive folder
            experiment_id (int, optional): id number for current experiment
        """

        print("Started archive at {0}".format(time.strftime("%Y_%m_%d %H%M%S")))
                   
        #set / refresh the model_args dictionary
        self.get_model_args()

        #template folder contains all of the appropriate, but empty, subfolders
        template_folder = os.path.join(self.archive_path , "_summary")

        #if the archive folder already exists, delete it
        if os.path.exists(archive_folder):
            rmtree(archive_folder, ignore_errors = True)

        # print (template_folder)
        # print (archive_folder)
        #create archive folder structure
        copytree(template_folder, archive_folder)

    
        #close instance of TransCAD
        if self.tc is not None:
            self.stop_transcad()

        print("Completed archive at {0}\n".format(time.strftime("%Y_%m_%d %H%M%S")))

    def __load_model(self, scen = 'Base'):
        """
        Create Model Object to set parameters and run
            scen = default to Base if in setup, pass scenario name if running
        """
        self.model_obj = self.tc.CreateGisdkObject("gis_ui", "Model.Runtime")
        modelFileName = os.path.join(self.model_path , 'CTPS_TDM23.model')
        self.model_obj.CloseModel()
        self.model_obj.SetModel({
            "Model": modelFileName,
            "Scenario": scen,
            "Silent": True
        })

    def start_transcad(self):
        """
        Launch TransCAD and call methods to reset scenario
        """
        # wack transCAD if it is open (cannot restart model)
        _logger.warn("[(re)starting TransCAD instance]")
        os.system("TASKKILL /F /IM tcw.exe")
        
        # now connect to TransCAD
        logf = os.path.abspath( self.model_path + "TC_log_" + format(time.strftime("%Y_%m_%d %H%M%S")) + ".txt" )
        self.tc = cp.TransCAD.connect(log_file = logf)
        
        if self.tc is None:
            _logger.error("ERROR: failed to attach to a TransCAD instance")
            sys.exit()

    def stop_transcad(self):
        """
        Close TransCAD
        """
        _logger.warn("Closing this instance of TransCAD")
        cp.TransCAD.disconnect()
        time.sleep(5)
        self.tc = None        

    def __direct_scenario_param(self, macro, ds_args, evar, expvars):
        # used when scope value directly matches model scenario parameter
        # no transformation needed

        param_name = ds_args
        param_value = expvars[evar]

        self.scenario_values[param_name] = param_value
        return [evar]
      
    # ============================================================================
    # Hooks to macros and methods for CTPS
    # ============================================================================

    # dictionary of method, macro and list of parameters by experiment variable
    # the first element of the parameter list is a list of input datasets passed to the macro
    # input datasets are referenced by their model argument names
    # __set_simple_evar is run for experiment variables that are applied to a single input dataset

    # direct set - i.e. no macro, only argument is the tdm23 parameter name
    __METHOD_BY_EVAR = {
        'Expanded Work from Home':      (__direct_scenario_param,None,"Regional Remote Level"),
        'PnR Max Shadow Cost':      (__direct_scenario_param,None,"emat_transit_pnr_max_factor")
    }


    __MEASURE_PARSERS = [

        TableParser(
                        filename = "wfh_calclog.csv",
			            measure_getters ={'Regional_total': iloc[0, 1],},
                        on_bad_lines = 'skip'

                    )
    ]