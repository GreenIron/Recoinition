# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 18:31:59 2016

@author: victor
"""
import csv
import cv2
import ast


class Template:
    def __init__(self,  pathInfo):
        """
        Import image
        """
        self.image = cv2.imread(pathInfo+".png", cv2.IMREAD_COLOR)
        # Import contextual information
        with open(pathInfo+".csv", 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == "path":
                    self.path = row[1]
                elif row[0] == "numismatic":
                    self.numismatic = row[1]
                elif row[0] == "radius":
                    self.radius = float(row[1])
                elif row[0] == "side_type":
                    self.side_type = row[1]
                elif row[0] == "relative_coord":
                    self.relative_coord = ast.literal_eval(row[1])
                elif row[0] == "center":
                    self.center = row[1]
                                    
    def update_template(self,  pathInfo):
        """
        Import image
        """
        self.image = cv2.imread(pathInfo+".png", cv2.IMREAD_COLOR)
        # Import contextual information
        with open(pathInfo+".csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == "path":
                    self.path = row[1]
                elif row[0] == "numismatic":
                    self.numismatic = row[1]
                elif row[0] == "radius":
                    self.radius = row[1]
                elif row[0] == "side_type":
                    self.side_type = row[1]
                elif row[0] == "relative_coord":
                    self.relative_coord = ast.literal_eval(row[1])
                elif row[0] == "center":
                    self.center = row[1]
