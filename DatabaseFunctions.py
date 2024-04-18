import os
import re
import yaml
import db_root as db
import numpy as np
import pandas as pd
import datetime as dt
import argparse
import datetime
import shutil
import pathlib
import sys
import TzFuncs
import time
import json
from HelperFunctions import sub_path



class DatabaseFunctions():

    def __init__(self,ini={}):
        self.splits = ' |,'
        for key,val in db.db_config['RootDirs'].items():
            setattr(self, key, val)
        self.db_ini = self.Database+'Calculation_Procedures/TraceAnalysis_ini/'
        print('Initialized using db_root: ', self.Database)
        # Read base config 
        with open(f'{self.db_ini}_config.yml') as f:
            self.ini = yaml.safe_load(f)
            print(f'Loaded {self.db_ini}_config.yml')

        # Read user provided configuration as a dictionary or .yml file        
        if type(ini)==str:
            if os.path.isfile(ini):
                with open(ini) as f:
                    yml_in = {os.path.basename(ini).split('.')[0]:yaml.safe_load(f)}
                    self.ini.update(yml_in)
                    print(f'Loaded {ini}')
            elif os.path.isfile(f'{self.db_ini}{ini}'):
                with open(f'{self.db_ini}{ini}') as f:
                    yml_in = {ini.split('.')[0]:yaml.safe_load(f)}
                    self.ini.update(yml_in)
                    print(f'Loaded {self.db_ini}{ini}')
                    
        elif type(ini)==dict:
            self.ini.update(ini)
            print(f'Loaded user provided settings:')
            print(ini)
        else:
            print('Provide properly formatted ini')
        self.find_Sites()

    def find_Sites(self):
        self.years_by_site = {}
        for f in os.listdir(self.db_ini):
            if f.startswith('_') == False:
                self.years_by_site[f] = []
        for y in os.listdir(self.Database):
            if y[0].isdigit():
                for site in self.years_by_site.keys():
                    if os.path.isdir(f'{self.Database}/{y}/{site}'):
                        self.years_by_site[site].append(y)

    def read_db(self,siteID,Years,stage,trace_names):
        tv_info = self.ini['Database']['Timestamp']
        tr_info = self.ini['Database']['Traces']
        tv = [np.fromfile(f'{self.Database}{y}/{siteID}/{stage}/{tv_info["name"]}',tv_info['dtype']) for y in Years]
        tv = np.concatenate(tv,axis=0)
        DT = pd.to_datetime(tv-tv_info['base'],unit=tv_info['base_unit']).round('S')
        traces={}        
        for f in trace_names:
            try:
                trace = [np.fromfile(f'{self.Database}{y}/{siteID}/{stage}/{f}',tr_info['dtype']) for y in Years]
                traces[f]=np.concatenate(trace,axis=0)
            except:
                traces[f]=np.empty(tv.shape)*np.nan
        self.data = pd.DataFrame(data=traces,index=DT)

    def dateIndex(self):
        dateCols = [i for i in re.split(self.splits,self.job['dateCols'].replace('\n',''))]
        if self.job['dateFmt'] == 'Auto':
            Date_col = dateCols[0]
            self.Data[Date_col] = pd.DatetimeIndex(self.Data[Date_col])
            self.Data = self.Data.set_index(Date_col)
        elif self.job['dateFmt'] is not None:
            self.Data['Timestamp'] = ''
            self.Data['Shift'] = False
            for col in dateCols:
                try:
                    ix = self.headerList.index(col)
                    unit = self.headerUnits[ix]
                    if unit.upper() == 'HHMM':
                        self.Data.loc[self.Data[col]==2400,[col,'Shift']]=[0,True]
                    self.Data['Timestamp'] = self.Data['Timestamp'].str.cat(self.Data[col].astype(str).str.zfill(len(unit)),sep='')
                except:
                    self.Data['Timestamp'] = self.Data['Timestamp'].str.cat(self.Data[col].astype(str),sep='')
            self.Data['Timestamp'] = pd.to_datetime(self.Data['Timestamp'],format=self.job['dateFmt'])
            self.Data.loc[self.Data['Shift'],'Timestamp']=self.Data.loc[self.Data['Shift'],'Timestamp']+pd.Timedelta(days=1)
            self.Data.drop('Shift',inplace=True,axis=1)
            self.Data = self.Data.set_index('Timestamp')
        if self.job['is_dst'] == 'True':
            lat_lon=[float(self.ini[self.siteID]['latitude']),float(self.ini[self.siteID]['longitude'])]
            tzf = TzFuncs.Tzfuncs(lat_lon=lat_lon,DST=True)
            tzf.convert(self.Data.index)
            self.Data = self.Data.set_index(tzf.Standard_Time)
        self.Aggregate()
        self.Data=self.Data.resample('30min').first()

    def Aggregate(self):
        if 'Aggregate' in self.job and self.job['Aggregate'] is not None:
            self.Data = self.Data.agg(re.split(self.splits,self.job['Aggregate'].replace('\n','')),axis=1)
            print(self.Data.head())

    def padFullYear(self):
        for self.year in self.Data.index.year.unique():
            self.byYear = pd.DataFrame(data={'Timestamp':pd.date_range(start = f'{self.year}01010030',end=f'{self.year+1}01010001',freq='30T')})
            self.byYear = self.byYear.set_index('Timestamp')
            self.byYear = self.byYear.join(self.Data)
            
            d_1970 = datetime.datetime(1970,1,1,0,0)
            self.byYear['Floor'] = self.byYear.index.floor('D')
            self.byYear['Secs'] = ((self.byYear.index-self.byYear['Floor']).dt.seconds/ (24.0 * 60.0 * 60.0))
            self.byYear['Days'] = ((self.byYear.index-d_1970).days+int(self.ini['Database']['Timestamp']['base']))

            self.byYear[self.ini['Database']['Timestamp']['name']] = self.byYear['Secs']+self.byYear['Days']
            self.byYear = self.byYear.drop(columns=['Floor','Secs','Days'])
            self.Write_Trace()

    def Write_Trace(self):
        class_dict = self.__dict__
        self.write_dir = sub_path(class_dict,f'{self.Database}/**year**/**siteID**/')+self.job['subfolder']
        if os.path.isdir(self.write_dir)==False:
            print('Creating new directory at:\n', self.write_dir)
            os.makedirs(self.write_dir)
        for T in self.byYear.columns:
            if T == self.ini['Database']['Timestamp']['name']:
                fmt = self.ini['Database']['Timestamp']['dtype']
            else:
                fmt = self.ini['Database']['Traces']['dtype']
            # try:
            Trace = self.byYear[T].astype(fmt).values
            del_chars = '()<>:"\|?'
            for c in del_chars:
                T = T.replace(c,'')
            T = T.replace('*','star').replace('/','_')
            if 'prefix' in self.job and self.job['prefix'] is not None and T != self.ini['Database']['Timestamp']['name']:
                T = self.job['prefix'] + '_' + T
            if 'suffix' in self.job and self.job['suffix'] is not None and T != self.ini['Database']['Timestamp']['name']:
                T += '_' + self.job['suffix']
            print(f'Writing: {self.write_dir}/{T}')
            with open(f'{self.write_dir}/{T}','wb') as out:
                Trace.tofile(out)
            # except:
            #     print(f'Could not write column: {T}')

    def copy_raw_data_files(self,dir=None,file=None,format='dat'):
        
        class_dict = self.__dict__
        copy_to = sub_path(class_dict,self.job['copy_to'])
        if os.path.isdir(copy_to) == False:
            print('Warning: ',copy_to,' Does not exist.  Ensure this is the correct location to save then create the folder before proceeding.')
            sys.exit()
        elif os.path.isdir(f"{copy_to}/{self.job['subfolder']}") == False:
            os.makedirs(f"{copy_to}/{self.job['subfolder']}")
        copy_to = f"{copy_to}/{self.job['subfolder']}"
        if format == 'dat':
            fname = pathlib.Path(dir+'/'+file)
            mod_time = datetime.datetime.fromtimestamp(fname.stat().st_mtime).strftime("%Y%m%dT%H%M")
            shutil.copy(f"{dir}/{file}",f"{copy_to}/{self.job}_{mod_time}.dat")
            with open(f"{copy_to}/{self.job}_README.md",'w+') as readme:
                str = f'# README\n\nLast update{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'
                str += '\n\n' +self.job['readme']
                readme.write(str)
        elif format == 'csv':
            file.to_csv(f"{copy_to}/{self.job}.csv")
            with open(f"{copy_to}/{self.job}_README.md",'w+') as readme:
                str = f'# README\n\nLast update{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'
                str += '\n\n' +self.job['readme']
                readme.write(str)

    def alignHeaderList(self,Data):
        cols_in = Data.columns
        if len(cols_in)<len(self.headerList):
            self.headerList = self.headerList[:len(cols_in)]
            self.headerUnits = self.headerUnits[:len(cols_in)]
        elif len(cols_in)>len(self.headerList):
            for i in range(len(self.headerList),len(cols_in)):
                self.headerList.append(cols_in[i])
                self.headerUnits.append(None)
        return(Data)
    
    def excludeCols(self):
        if 'exclude' in self.job and self.job['exclude'] is not None:
            colFilter = [cf for cf in re.split(self.splits,self.job['exclude'].replace('\n','')) if cf in self.Data.columns]
            self.Metadata.drop(colFilter,inplace=True,axis=1) 
            self.Data.drop(colFilter,inplace=True,axis=1)
        self.Data.dropna(axis=1,inplace=True,how='all')
        drp = [c for c in self.Data.columns if type(c) is not str]
        self.Data.drop(drp,inplace=True,axis=1)

class MakeCSV(DatabaseFunctions):
    def __init__(self,Sites=None,Years=[dt.datetime.now().year],ini={}):
        super().__init__(ini)
        T1 = time.time()
        if Sites is None:
            Sites = self.years_by_site.keys()
        for self.siteID in Sites:
            for req in ini:
                req = req.split('.')[0]
                stage = self.ini[req]["stage"]
                traces = list(self.ini[req]['Traces'].keys())
                print(f'Creating {req} for {self.siteID}')
                self.read_db(self.siteID,Years,stage,traces)
                if 'by_year' in self.ini['data'].keys() and self.ini[req]['by_year']:
                    Start = self.data.index-pd.Timedelta(30,'m')
                    for self.Year in Years:
                        self.write_csv(self.data.loc[Start.year==self.Year].copy(),self.ini[req])
                else:
                    self.write_csv(self.data,self.ini[req])
                    
    def write_csv(self,df,config):
        if 'output_path' not in config.keys():
            config['output_path']=os.getcwd()+'/'
        if 'timestamp' not in config.keys():
            config['timestamp']={'output_name': 'TIMESTAMP',
                                'timestamp_fmt': '%Y-%m-%d %H%M',
                                'timestamp_units': 'yyyy-mm-dd HHMM'}
        if 'units_in_header' not in config.keys():
            config['units_in_header'] = False
        if df.empty:
            print(f'No data to write for {self.siteID}: {self.Year}')
        else:
            df[config['timestamp']['output_name']] = df.index.floor('Min').strftime(config['timestamp']['timestamp_fmt'])
            class_dict = self.__dict__
            output_path = sub_path(class_dict,config['output_path'])
            if os.path.exists(output_path)==False:
                os.makedirs(output_path)
            output_path = sub_path(class_dict,config['output_path'])
            print(output_path)
            if config['units_in_header'] == True:
                unitDict = {key:config['Traces'][key]['Units'] for key in config['Traces'].keys()}
                unitDict[config['timestamp']['output_name']] = config['timestamp']['timestamp_units']
                df = pd.concat([pd.DataFrame(index=[-1],data=unitDict),df])
            df=df.fillna(config['na_value'])
            df.to_csv(output_path,index=False)
        
            

class MakeTraces(DatabaseFunctions):
    # Accepts an ini file that prompt a search of the datadump folder - or a pandas dataframe with a datetime index
    def __init__(self,ini='ini_files/WriteTraces.yml',DataTable=None):
        print()
        super().__init__(ini)
        jobs = os.path.basename(ini.split('.')[0])   
        if DataTable is None:
            for task,self.job in self.ini[jobs].items():
                print('Processing: ',task,self.job)
                self.siteID = self.job['siteID']
                self.findFiles()
        else:
            self.job = self.ini[jobs].keys()[0]
            self.siteID = self.job['Site']
            self.Data = DataTable
            self.Process()
    
    def Process(self):
        self.dateIndex()
        self.excludeCols()
        self.padFullYear()


    def findFiles(self):
        self.Data = pd.DataFrame()
        self.Metadata = pd.DataFrame()
        class_dict = self.__dict__
        search_dir = sub_path(class_dict,self.job['inputData'])
        if 'subtable' in self.job and self.job['subtable'] is not None:
            self.SubTables = {k:{} for k in ['Data','Metadata']}
        # Search relevant directory for input data
        for dir,_,files in os.walk(search_dir):
            for file in (files):
                fn = f"{dir}/{file}"                
                if len([p for p in self.job['fileNamePatterns'].replace('\n','').split(' ') if p not in fn])==0:                    
                    if 'copy_to' in self.job == 'True':
                        self.copy_raw_data_files(dir=dir,file=file)
                    print(fn)
                    if hasattr(self,'SubTables'):
                        self.readSubTables(fn)
                    else:
                        self.readSingle(fn)
            
        if hasattr(self,'SubTables'):
            for k in self.SubTables['Data'].keys():
                self.Data = self.SubTables['Data'][k]
                self.Metadata = self.SubTables['Metadata'][k]
                self.Process()
        else:
            self.Process

    def readSingle(self,fn):
        if 'headerRow' in self.job and self.job['headerRow'] is not None:
            header = pd.read_csv(fn,skiprows=int(self.job['headerRow']),nrows=int(self.job['firstDataRow'])-int(self.job['headerRow']))
            self.Metadata = pd.concat([self.Metadata,header],axis=0)
            self.headerList = header.columns
        else:
            self.headerList = self.job['headerList'].replace('\n','').split(' ')
            self.headerUnits = self.job['headerUnits'].replace('\n','').split(' ')
            header = pd.DataFrame(columns=self.headerList)
            header.iloc[0] = self.headerUnits
            self.Metadata = pd.concat([self.Metadata,header],axis=0)
        Data = pd.read_csv(fn,skiprows=int(self.job['firstDataRow']),header=None)
        Data.columns=self.headerList
        self.Data = pd.concat([self.Data,Data],axis=0)

    def readSubTables(self,fn):
        try:
            Data = pd.read_csv(fn,header=None,na_values=[-6999,6999])
        except:
            try:
                Data = pd.read_csv(fn,header=None,na_values=[-6999,6999],skiprows=1)
                First = pd.read_csv(fn,header=None,na_values=[-6999,6999],nrows=1)
            except:
                print(f'\n\nWarinng: Could not read {fn} Check for errors.  It it is a cr10x file, make sure all columns are present.  Outputs from two programs in one file will cuase problems!\n\n')
            pass
        for key,subtable in self.job['subtable'].items():
            if key not in self.SubTables:
                self.SubTables['Data'][key] = pd.DataFrame()
                self.SubTables['Metadata'][key] = pd.DataFrame()
            self.headerList = list(subtable['headerList'].keys())
            self.headerUnits = list(subtable['headerList'].values())
            Data = self.alignHeaderList(Data)
            header = pd.DataFrame(columns=self.headerList,data=[self.headerUnits],index=[0])
            header.iloc[0] = self.headerUnits
            
            self.col_num = self.headerList.index('ID')
            Subtable = Data.loc[Data[self.col_num]==subtable['ID']]
            Subtable = Subtable[Subtable.columns[0:len(self.headerList)]]
            self.SubTables['Metadata'][key] = pd.concat([self.SubTables['Metadata'][key],header],axis=0)
            Subtable.columns=header.columns
            self.SubTables['Data'][key] = pd.concat([self.SubTables['Data'][key],Subtable],axis=0)
        
class GSheetDump(DatabaseFunctions):
    def __init__(self, ini='ini_files/WriteTraces_Gsheets.ini'):
        super().__init__(ini)
        jobs = os.path.basename(ini.split('.')[0])
        for task,self.job in self.ini[jobs].items():
            print('Processing: ',task)
            self.siteID = self.job['siteID']
            self.readSheet()

    def readSheet(self):
        k = list(self.job['subtable'].keys())
        subtable = self.job['subtable'][k[0]]
        self.Data = pd.read_html(self.job['filename'],
                     skiprows=int(self.job['headerRow']))[int(subtable['ID'])]
        if 'headerList' in subtable and subtable['headerList'] is not None:
            self.headerList = list(subtable['headerList'].keys())
            self.headerUnits = list(subtable['headerList'].values())
            print(self.Data.columns)
            self.Data = self.alignHeaderList(self.Data)            
            self.Data.columns=self.headerList  
        else:
            self.headerList = self.Data.columns
            self.headerUnits = [None for i in self.headerList]
        self.Metadata=pd.DataFrame(data=[self.headerUnits],columns=self.headerList)
        if 'copy_to' in self.job == 'True':
            self.copy_raw_data_files(file=self.Data,format='csv')
        self.dateIndex()        
        self.excludeCols()
        self.padFullYear()


if __name__ == '__main__':
    T1 = time.time()
    file_path = os.path.split(__file__)[0]
    os.chdir(file_path)

    CLI=argparse.ArgumentParser()

    CLI.add_argument(
        "--Task",
        nargs='*',
        type=str,
        default=['Help'],
        )
    
    CLI.add_argument(
        "--Sites",
        nargs='*',
        type=str,
        default=None,
        )
    
    CLI.add_argument(
        "--Years",
        nargs='*',
        type=int,
        default=None,
        )
    
    CLI.add_argument(
        "--ini",
        nargs='*',
        type=str,
        default=None,
        )
        
    args = CLI.parse_args()
    # if args.Task == 'Help' or 'Help' in args.Task:
    
    ini_defaults = {
        'Help':'N\A',
        'CSVDump':'ini_files/Write_CSV_Files.yml',
        'Write':'ini_files/WriteTraces.yml',
        'GSheetDump':'ini_files/WriteTraces_Gsheets.yml'
    }
    
    for i,Task in enumerate(args.Task):
        print(Task)
        if args.ini is None:
            ini = ini_defaults[Task]
        else:
            ini = args.ini
        if args.Years is not None:
            Years = np.arange(min(args.Years),max(args.Years)+1)
        else:
            Years = None

        if Task == 'CSVDump':
            MakeCSV(args.Sites,Years,ini=ini)
        elif Task == 'Write':
            MakeTraces(ini=ini)
        elif Task == 'GSheetDump':
            GSheetDump(ini=ini)
        elif Task == 'Help':
            print('Help: \n')
            print("--Task: options ('CSVDump', 'Write', or 'GSheetDump')")
            print("--Sites: Leave blank to run all sites or give a list of sites delimited by spaces, e.g., --Sites BB BB2 BBS )\n Only applies if Task == Read")
            print("--Years: Leave blank to run all years or give a list of years delimited by spaces, e.g., --Years 2020 2021 )\n Only applies if Task == Read")
            print("--ini: Leave blank to run default or give a list of ini files corresponding to each Task")
    print('Request completed.  Time elapsed: ',np.round(time.time()-T1,2),' seconds')