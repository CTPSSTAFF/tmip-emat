from typing import List, Union, Mapping
import yaml
import warnings
import os, sys, time
from shutil import copyfile, copy, rmtree, copytree
import glob
import pandas as pd
import numpy as np
import subprocess

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
        
        self.model_path     = self.config['model_path']
        #self.model_path     = os.path.normpath(self.config['model_path']) 
        self.archive_path   = self.config['model_archive']
        self.post_proc      = self.config['post_processor']
        self.log_path   = self.config['log_path']
        #self.archive_path   = os.path.normpath(self.config['model_archive'])
        self.parent_scen    = self.config['parent_scen']
        self.scen_name      = 'emat'
        self.rel_output_path = os.path.join("outputs",self.parent_scen, self.scen_name)
        self.scenario       = os.path.join(self.parent_scen, self.scen_name)

        # derived values based on tdm23 structure
        self.tdm_ui         = self.model_path + '/ui/tdm23_ui.dbd'
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
            print("Setting experiment variable for " + str(xv) + " to " + str(expvars[xv]))


            try:
                func, macro, args = self.__METHOD_BY_EVAR[xv]
                variables_done += func(self, macro, args, xv, expvars)
                      
            except KeyError:
                _logger.info("Experiment variable method not found, defaulting to direct set")
                self.scenario_values[xv] = expvars[xv]
                variables_done += xv
        
        # set values to scenario
        self.load_model()        
        self.model_obj.DeleteScenario(self.scenario)           
        self.model_obj.CreateScenario(self.parent_scen, self.scen_name, "emat scenario")     
        self.model_obj.SetScenario(self.scenario)
        self.model_obj.SetScenarioValues(self.scenario_values)

        print("Completed model setup at {0}\n".format(time.strftime("%Y_%m_%d %H%M%S")))


    def pre_process(self):
        """
        delete scenario output folder
        """
        #delete any existing model reports and logs from the output folder
        if self.config["debug_nodelete"] == False:
            rmtree(self.scenario_path, ignore_errors=True)

    
    def run(self):
        """
        Run the model.

        TransCAD should already have been started and inputs set with the 'pre_process' method.
        
        """

        #start transcad if it isn't already running
        if self.tc is None:
            self.start_transcad()

        # load model
        self.load_model(scen = self.scenario)
            
        print("Starting model run at {0}".format(time.strftime("%Y_%m_%d %H%M%S")))
        _logger.info("Starting model run at {0}".format(time.strftime("%Y_%m_%d %H%M%S")))

        # try running tdm23 without threading
        #self.model_obj.RunModel({})
        
        
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
                (optional) Path to model archive folder
                defaults to emat scenario outputs

            Raises:
                KeyError:
                    If post process macro is not available for the specified measure
        """

        # output_path is specified when running against archive, otherwise run 
        # default scenario
        if output_path is None: 
            output_path = os.path.normpath(self.scenario_path)

        # run generic post processor
        result = subprocess.run([self.post_proc,
                                 output_path],
                                 shell=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)

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

        print("Archiving results to {0} at {1}".format(archive_folder, time.strftime("%Y_%m_%d %H%M%S")))

        # create output folder
        if not os.path.exists(archive_folder):
            os.makedirs(archive_folder)
            time.sleep(2)   

        # copy full folders
        full_folder = ["_summary","_postproc","_networks","_assignment","_skim"]
        for folder in full_folder:
            source = os.path.join(self.scenario_path, folder)
            dest = os.path.join(archive_folder, folder)
            rmtree(dest,ignore_errors=True)
            copytree(source, dest) 

        # copy scenario file and reports
        copy(os.path.join(self.model_path, "CTPS_TDM23.scenarios"), archive_folder)       
        copy(os.path.join(self.scenario_path, "tdm23.db"), archive_folder)       
        # Issue copying with wildcards
        #copy(os.path.join(self.scenario_path, "*.xml"), archive_folder)       
        #copy(os.path.join(self.scenario_path, "*.xsl"), archive_folder)       
        copy(os.path.join(self.scenario_path, "config.json"), archive_folder)       

        #close instance of TransCAD
        if self.tc is not None:
            self.stop_transcad()

        print("Completed archive at {0}\n".format(time.strftime("%Y_%m_%d %H%M%S")))

    def load_model(self, scen = 'Base'):
        """
        Create Model Object to set parameters and run
            scen = default to Base if in setup, pass scenario name if running
        """
        self.model_obj = self.tc.CreateGisdkObject(self.tdm_ui, "Model.Runtime")
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
        logf = os.path.abspath( self.log_path + "\\TC_log_" + format(time.strftime("%Y_%m_%d%H%M%S")) + ".txt" )
        self.tc = cp.TransCAD.connect(log_file = logf)
        print("Log file {0}\n".format(logf))

        # reset transcad report and log files (to avoid overfilling)
        self.tc.ResetReportFile()
        self.tc.ResetLogFile()
        
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
    
    def __network_param(self, macro, ds_args, evar, expvars):
        # used to set linksnodes, routesstops, and associated files

        param_value = expvars[evar]

        if param_value == '50pln':
            netbase = "%InputFolder%\\networks\\tdm23.1.0\\2050pln\\tdm23_1_0_2050pln" 
            modetab = "%InputFolder%\\params\\transit_modes_2050_20231231.bin"
            warmspd = "%InputFolder%\\networks\\tdm23.1.0\\2050pln\\warm_start\\"
        elif param_value == '19base':
            netbase = "%InputFolder%\\networks\\v3\\2019\\tdm23_1_0_v3_2019"
            modetab = "%InputFolder%\\params\\transit_modes_2019_20250114_ovttweights.bin"
            warmspd = "%InputFolder%\\networks\\v3\\2019\\warm_start\\"
        else: 
            _logger.error(f"ERROR: network scenario {param_value} undefined")
            _logger.error(f"evar {evar}")
            _logger.error(f"expvars {expvars}")
            sys.exit()

        self.scenario_values["Highway Input"] = netbase + "_ln.dbd"
        self.scenario_values["Transit Input"] = netbase + "_rs.rts"
        self.scenario_values["Turn Penalties"] = netbase + "_tpen.bin"
        self.scenario_values["Init Speeds - am"] = warmspd + "base_am_speeds.bin"
        self.scenario_values["Init Speeds - md"] = warmspd + "base_md_speeds.bin"
        self.scenario_values["Init PnR Demand - am"] = warmspd + "pnr_ta_acc_am.mtx"
        self.scenario_values["Transit Mode Table"] = modetab

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
        "Dry Run":                  (__direct_scenario_param,None,"DryRun"),
        "Electric Bike":            (__direct_scenario_param,None, "Bike Speed"),
        "TNC Availability":         (__direct_scenario_param,None, "TNC Fare Wait Adjustment"),
        "HRT Reliability":          (__direct_scenario_param,None, "Transit HRT Time Adjustment"),
        "Post-Pandemic WFH":        (__direct_scenario_param,None, "WFH Adjustment"),
        "AV Operations":            (__direct_scenario_param,None, "AV PCE Adjustment"),
        "Highway Toll":             (__direct_scenario_param,None, "Highway Toll Adjustment"),
        "Transit Fare":             (__direct_scenario_param,None, "Transit Fare Adjustment"),
        "Bus Lanes":                (__direct_scenario_param,None, "Enable Bus Lanes"),
        "Network Scenario":         (__network_param,None, "Network Scenario"),
    }


    __MEASURE_PARSERS = [

        TableParser(
                        filename = "_summary\\wfh_calclog.csv",
			            measure_getters ={'Regional_total': iloc[0, 1],},
                        on_bad_lines = 'skip'
                    ),

        TableParser(
                        filename = "_summary\\emat\\va.csv",
			            measure_getters ={
                            'BRMPO_VA_ZV_Shr': loc['BRMPO', 'zv_p'],
                            'BRMPO_VA_IV_Shr': loc['BRMPO', 'iv_p'],
                            'BRMPO_VA_SV_Shr': loc['BRMPO', 'sv_p'],
                            },
                        on_bad_lines = 'skip',
                        index_col=0
                    ),

        TableParser(
                        filename = "_summary\\emat\\per_trips.csv",
			            measure_getters ={
                            'BRMPO_Auto_Shr': loc['BRMPO', 'auto_p'],
                            'BRMPO_NM_Shr': loc['BRMPO', 'nonm_p'],
                            'BRMPO_TRN_Shr': loc['BRMPO', 'trn_p'],
                            'BRMPO_SB_Shr': loc['BRMPO', 'sb_p'],
                            'BRMPO_SB_Trip': loc['BRMPO', 'sb'],
                            },
                        on_bad_lines = 'skip',
                        index_col=0
                    ),
                    
        TableParser(
                        filename = "_summary\\emat\\hh_trips_geo.csv",
			            measure_getters ={
                            'BRMPO_hbw': loc['BRMPO', 'hbw'],
                            'BRMPO_hbnw': loc['BRMPO', 'hbnw'],
                            'BRMPO_nhbw': loc['BRMPO', 'nhbw'],
                            'BRMPO_nhbnw': loc['BRMPO', 'nhbnw'],
                            'BRMPO_HH_Trips': loc['BRMPO', 'Total'],
                            },
                        on_bad_lines = 'skip',
                        index_col=0
                    ),

        TableParser(
                        filename = "_summary\\emat\\veh_trips.csv",
			            measure_getters ={
                            'BRMPO_Auto_Trips': loc['BRMPO', 'auto'],
                            'BRMPO_DA_Trips': loc['BRMPO', 'da'],
                            'BRMPO_SR_Trips': loc['BRMPO', 'sr'],
                            'BRMPO_MTRK_Trips': loc['BRMPO', 'mtrk'],
                            'BRMPO_HTRK_Trips': loc['BRMPO', 'htrk'],
                            },
                        on_bad_lines = 'skip',
                        index_col=0
                    ),          

        TableParser(
                        filename = "_summary\\emat\\vmt_factype.csv",
			            measure_getters ={
                            'BRMPO_Freeway_VMT': loc['BRMPO', 'Freeway'],
                            'BRMPO_Expressway_VMT': loc['BRMPO', 'Expressway'],
                            'BRMPO_Mj_Arterial_VMT': loc['BRMPO', 'Major Arterial'],
                            'BRMPO_Mn_Arterial_VMT': loc['BRMPO', 'Minor Arterial'],
                            'BRMPO_Total_VMT': loc['BRMPO', 'Total'],
                            },
                        on_bad_lines = 'skip',
                        index_col=0
                    ),               

        TableParser(
                        filename = "_summary\\emat\\vmt_mode.csv",
			            measure_getters ={
                            'BRMPO_Auto_VMT': loc['BRMPO', 'auto_vmt'],
                            'BRMPO_DA_VMT': loc['BRMPO', 'da_vmt'],
                            'BRMPO_SR_VMT': loc['BRMPO', 'sr_vmt'],
                            'BRMPO_MTRK_VMT': loc['BRMPO', 'mtrk_vmt'],
                            'BRMPO_HTRK_VMT': loc['BRMPO', 'htrk_vmt'],
                            },
                        on_bad_lines = 'skip',
                        index_col=0
                    ),     

        TableParser(
                        filename = "_summary\\emat\\cvmt_factype.csv",
			            measure_getters ={
                            'BRMPO_Freeway_CVMT': loc['BRMPO', 'Freeway'],
                            'BRMPO_Expressway_CVMT': loc['BRMPO', 'Expressway'],
                            'BRMPO_Mj_Arterial_CVMT': loc['BRMPO', 'Major Arterial'],
                            'BRMPO_Mn_Arterial_CVMT': loc['BRMPO', 'Minor Arterial'],
                            'BRMPO_Total_CVMT': loc['BRMPO', 'Total'],
                            },
                        on_bad_lines = 'skip',
                        index_col=0
                    ),       

        TableParser(
                        filename = "_summary\\emat\\cvmt_factype.csv",
			            measure_getters ={
                            'BRMPO_Freeway_CVMT': loc['BRMPO', 'Freeway'],
                            'BRMPO_Expressway_CVMT': loc['BRMPO', 'Expressway'],
                            'BRMPO_Mj Arterial_CVMT': loc['BRMPO', 'Major Arterial'],
                            'BRMPO_Mn Arterial_CVMT': loc['BRMPO', 'Minor Arterial'],
                            'BRMPO_Total_CVMT': loc['BRMPO', 'Total'],
                            },
                        on_bad_lines = 'skip',
                        index_col=0
                    ),        

        TableParser(
                        filename = "_summary\\emat\\trn_mode.csv",
			            measure_getters ={
                            'lbus': loc['lbus',:],
                            'xbus': loc['xbus',:],
                            'brt': loc['brt',:],
                            'lrt': loc['lrt',:],
                            'hr': loc['hr',:],
                            'cr': loc['cr',:],
                            'bt': loc['bt',:],
                            'shtl': loc['shtl',:],
                            'rta': loc['rta',:],
                            'regb': loc['regb',:],
                            'total_transit': loc['Total',:]
                            },
                        on_bad_lines = 'skip',
                        index_col=0
                    ),                                                                                          
        TableParser(
                        filename = "_summary\\emat\\emission_mpo.csv",
			            measure_getters ={
                            'BRMPO_CO2': loc['BRMPO', 'CO2<br>(kg)'],
                            'BRMPO_CO': loc['BRMPO', 'CO<br>(kg)'],
                            'BRMPO_SO': loc['BRMPO', 'SO<br>(kg)'],
                            'BRMPO_NO': loc['BRMPO', 'NO<br>(kg)'],
                            'BRMPO_VOC': loc['BRMPO', 'VOC<br>(kg)'],
                            },
                        index_col=0
                    ),    
        TableParser(
                        filename = "_summary\\postproc\\equity_aggregated.csv",
                        measure_getters ={
                            'linc_jobs_hwy': loc['Low-income', 'jobs_hwy'],
                            'linc_jobs_trn': loc['Low-income', 'jobs_trn'],
                            'linc_avg_time_hwy': loc['Low-income', 'avg_time_hwy'],
                            'linc_avg_time_trn': loc['Low-income', 'avg_time_trn'],
                            'linc_hlth_hwy': loc['Low-income', 'hlth_hwy'],
                            'linc_hlth_trn': loc['Low-income', 'hlth_trn'],
                            'linc_park_hwy': loc['Low-income', 'park_hwy'],
                            'linc_park_trn': loc['Low-income', 'park_trn'],
                            'mnr_jobs_hwy': loc['Minority', 'jobs_hwy'],
                            'mnr_jobs_trn': loc['Minority', 'jobs_trn'],
                            'mnr_avg_time_hwy': loc['Minority', 'avg_time_hwy'],
                            'mnr_avg_time_trn': loc['Minority', 'avg_time_trn'],
                            'mnr_hlth_hwy': loc['Minority', 'hlth_hwy'],
                            'mnr_hlth_trn': loc['Minority', 'hlth_trn'],
                            'mnr_park_hwy': loc['Minority', 'park_hwy'],
                            'mnr_park_trn': loc['Minority', 'park_trn'],
                            'n_linc_jobs_hwy': loc['Non-low-income', 'jobs_hwy'],
                            'n_linc_jobs_trn': loc['Non-low-income', 'jobs_trn'],
                            'n_linc_avg_time_hwy': loc['Non-low-income', 'avg_time_hwy'],
                            'n_linc_avg_time_trn': loc['Non-low-income', 'avg_time_trn'],
                            'n_linc_hlth_hwy': loc['Non-low-income', 'hlth_hwy'],
                            'n_linc_hlth_trn': loc['Non-low-income', 'hlth_trn'],
                            'n_linc_park_hwy': loc['Non-low-income', 'park_hwy'],
                            'n_linc_park_trn': loc['Non-low-income', 'park_trn'],
                            'n_mnr_jobs_hwy': loc['Nonminority', 'jobs_hwy'],
                            'n_mnr_jobs_trn': loc['Nonminority', 'jobs_trn'],
                            'n_mnr_avg_time_hwy': loc['Nonminority', 'avg_time_hwy'],
                            'n_mnr_avg_time_trn': loc['Nonminority', 'avg_time_trn'],
                            'n_mnr_hlth_hwy': loc['Nonminority', 'hlth_hwy'],
                            'n_mnr_hlth_trn': loc['Nonminority', 'hlth_trn'],
                            'n_mnr_park_hwy': loc['Nonminority', 'park_hwy'],
                            'n_mnr_park_trn': loc['Nonminority', 'park_trn'],
                        },
                        index_col=0
                    ),
        TableParser(
                        filename = "_summary\\postproc\\equity_dod.csv",
                        measure_getters ={
                            'inc_jobs_hwy': loc['income', 'jobs_hwy'],
                            'inc_jobs_trn': loc['income', 'jobs_trn'],
                            'inc_avg_time_hwy': loc['income', 'avg_time_hwy'],
                            'inc_avg_time_trn': loc['income', 'avg_time_trn'],
                            'inc_hlth_hwy': loc['income', 'hlth_hwy'],
                            'inc_hlth_trn': loc['income', 'hlth_trn'],
                            'inc_park_hwy': loc['income', 'park_hwy'],
                            'inc_park_trn': loc['income', 'park_trn'],
                            'eth_jobs_hwy': loc['minority', 'jobs_hwy'],
                            'eth_jobs_trn': loc['minority', 'jobs_trn'],
                            'eth_avg_time_hwy': loc['minority', 'avg_time_hwy'],
                            'eth_avg_time_trn': loc['minority', 'avg_time_trn'],
                            'eth_hlth_hwy': loc['minority', 'hlth_hwy'],
                            'eth_hlth_trn': loc['minority', 'hlth_trn'],
                            'eth_park_hwy': loc['minority', 'park_hwy'],
                            'eth_park_trn': loc['minority', 'park_trn'],
                        },
                        index_col=0
                    )



    ]