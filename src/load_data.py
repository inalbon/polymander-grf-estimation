"""
Created on Mon Oct 24 11:09:06 2022

@author: Malika In-Albon
"""

import os
from os.path import dirname, realpath, join

from log_force_plates import LogForcePlates
from log_polymander import LogPolymander


class LoadData:
    def __init__(self):
        """
        Class LoadData reads through a directory and creates one list containing
        the polymander object and one list containing force plate objects
        :param dir_path: directory path of the file to be loaded
        :param log_name: name of the file to be loaded
        """
        print('------------------------ load data -------------------------')
        self.list_polymander = []
        self.list_force_plates = []

    def load_polymander_data(self, dir_name, log_name=None):
        """
        Load a folder or a file of polymander data. Generate a list of
        LogPolymander objects
        :param dir_name: directory name
        :param log_name: file name
        """
        dir_path = join(f"{dirname(realpath(__file__))}/{dir_name}")
        if log_name is None:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".csv"):
                        self.list_polymander.append(LogPolymander(root, file))
            print(f'Folder {dir_name} has been loaded')
        else:
            log_name = f'{log_name}.csv'
            self.list_polymander.append(LogPolymander(dir_path, log_name))
            print(f'File {log_name} in folder {dir_name} has been loaded')

    def load_force_plates_data(self, dir_name, log_name=None):
        """
        Load a folder or a file of force plate data. Generate a list of
        LogForcePlates objects
        :param dir_name: directory name
        :param log_name: file name
        """
        dir_path = join(f"{dirname(realpath(__file__))}/{dir_name}")
        if log_name is None:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".txt"):
                        self.list_force_plates.append(LogForcePlates(root, file))
            print(f'Folder {dir_name} has been loaded')
        else:
            log_name = f'{log_name}.txt'
            self.list_force_plates.append(LogForcePlates(dir_path, log_name))
            print(f'File {log_name} in folder {dir_name} has been loaded')
