import numpy as np
import pandas as pd
import copy as cp
import statistics
import re
import math
import matplotlib.pyplot as plt 
from itertools import combinations
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import matplotlib.pyplot as plt
from  collections import Counter
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import time
import csv
from sklearn.linear_model import LinearRegression
from datetime import datetime
import random
import squarify
import seaborn as sns
import gradio as gr
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import psutil
import io

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


class Preprocessing:
    def __init__(self, dataset, dataFrame):
        self.dataset = dataset
        self.dataFrame = dataFrame   
        numeric_columns = self.dataFrame.select_dtypes(include=['int', 'float']).columns.tolist() # column label
        self.numeric_columns = [self.dataFrame.columns.get_loc(col) for col in numeric_columns]
    

    def val_manquante(self, attribute):
        L=[]
        for i in range(0,len(self.dataset[:,attribute])):
            if not re.fullmatch(r"\d+\.(:?\d+)?", str(self.dataset[i, attribute])):
                L.append(i)
        return L
    
    def calcul_mediane(self, attribute):
        datasetCurrated=np.delete(self.dataset[:,attribute], self.val_manquante(attribute))
        liste = cp.deepcopy(datasetCurrated)
        liste.sort()
        if liste.size % 2 !=0 :
        
            mediane=liste[((liste.size+1)//2) -1]
        else :
            mediane=(liste[(liste.size//2)-1]+liste[liste.size//2])/2
        return mediane
    
    def quartilles_homeMade(self, attribute):
        datasetCurrated=np.delete(self.dataset[:,attribute], self.val_manquante(attribute))
        liste = cp.deepcopy(datasetCurrated)
        liste.sort()
        q0=liste[0]
        q1=(liste[liste.size//4-1]+liste[liste.size//4]) /2
        q3=(liste[liste.size*3//4-1]+liste[liste.size*3//4]) /2
        q2=self.calcul_mediane(attribute)
        q4=liste[-1]
        return [q0,q1,q2,q3,q4]

    def Discretisation(self, attribute):
        vals = self.dataset[:,attribute].copy()
        vals.sort()
        q = 1+(10/3)*np.log10(self.dataset.shape[0])
        nbrelmt=math.ceil(self.dataset[:,attribute].shape[0]/q)
        
        for  val in range(0,self.dataset[:,attribute].shape[0]):  
            for i in range(0,vals.shape[0],nbrelmt):
                if(vals[i]>self.dataset[val,attribute]):
                    sup=i
                    break
            self.dataset[val,attribute]=np.median(vals[sup-nbrelmt:sup])
                
    def remplacement_val_manquantes(self, methode, attribute):
        missing=self.val_manquante(attribute)
        for i in missing:
            if methode=='Mode':
                self.dataset[i,attribute]= statistics.mode(self.dataset[:,attribute])    
            else:
                self.dataset[i,attribute]= np.mean([self.dataset[j,attribute] for j in range(0,len(self.dataset)) if self.dataset[j,-1]==self.dataset[i,-1] and not j in missing])

    def remplacement_val_aberrantes(self, methode,attribute):
        abberante=[]
        if methode=='Linear Regression':
            IQR=(np.percentile(self.dataset[:, attribute], 75)-np.percentile(self.dataset[:, attribute], 25))*1.5
            for i in range(0,len(self.dataset[:,attribute])):
                if (self.dataset[i,attribute] >(np.percentile(self.dataset[:, attribute], 75)+IQR) or self.dataset[i,attribute]<(np.percentile(self.dataset[:, attribute], 25)-IQR)):
                    abberante.append(i)
            X = np.delete(self.dataset, attribute, axis=1)
            X = np.delete(X, abberante, axis=0)
            y=self.dataset[:,attribute]
            y= np.delete(y, abberante, axis=0).reshape(-1, 1)

            model = LinearRegression().fit(X, y)
            
            for i in abberante:
                x2=np.delete(self.dataset, attribute, axis=1)
                X_new =x2[i,:].T.reshape(1, -1)
                self.dataset[i,attribute]=model.predict(X_new)[0][0]
        else:
            self.Discretisation(attribute)

    def remplacement_manquant_generale(self, methode):
        for i in range(0,self.dataset.shape[1]):
            self.remplacement_val_manquantes(methode,i) 

    def remplacement_aberantes_generale(self, methode):
        for i in range(0,self.dataset.shape[1]-1):
            self.remplacement_val_aberrantes(methode,i)
     
    def normalisation(self, methode, attribute, vmin, vmax):
        if methode=='Vmin-Vmax':
            vminOld=self.dataset[:,attribute].min()
            vmaxOld=self.dataset[:,attribute].max()
            for  val in range(0,self.dataset[:,attribute].shape[0]):
                self.dataset[val,attribute]=vmin+(vmax-vmin)*((self.dataset[val,attribute]-vminOld)/(vmaxOld-vminOld))
        else:
            vmean=np.mean(self.dataset[:,attribute])
            s=np.mean( (self.dataset[:,attribute]  -vmean)**2)
            for  val in range(0,self.dataset[:,attribute].shape[0]):
                self.dataset[val,attribute]=(self.dataset[val,attribute]-vmean)/s 
    
    def normalisation_generale(self, methode, vmin, vmax):
        for i in range(0,self.dataset.shape[1]):
            self.normalisation(methode,i, vmin, vmax)

    def preprocessing_general1(self, manque_meth, aberrante_meth, normalization_meth, vmin, vmax):
        self.remplacement_manquant_generale(manque_meth)
        self.remplacement_aberantes_generale(aberrante_meth)
        self.normalisation_generale(normalization_meth, int(vmin), int(vmax)) 

        return self.dataset
    
    #===============================DATASET2================================================================================================================================================================================================================================
    def year_mapping(self, time_period):
        self.dataFrame['Start date'] = pd.to_datetime(self.dataFrame['Start date'], errors='coerce')
        self.dataFrame['end date'] = pd.to_datetime(self.dataFrame['end date'], errors='coerce')

        yearly_intervals = self.dataFrame.groupby((self.dataFrame['Start date'].dt.year))['time_period'].agg(['min', 'max'])

        year_mapping = {}

        for year, interval in yearly_intervals.iterrows():
            year_mapping[(interval['min'], interval['max'])] = int(year)

        for interval, y in year_mapping.items():
            if interval[0] <= int(time_period) <= interval[1]:
                return y
        
    def convert_date(self, time_period, date):
        date = str(date)
        dd_mm_yy = re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b')
        dd_mmm = re.compile(r'\b\d{1,2}-[a-zA-Z]{3}\b')

        if dd_mm_yy.match(date):
            formatted_date = datetime.strptime(date, '%m/%d/%Y')
            return np.datetime64(formatted_date)
        elif dd_mmm.match(date):
            day, month = date.split('-')
            month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
            month = month_dict[month]
            year = self.year_mapping(time_period)
            return np.datetime64(datetime(int(year), month, int(day)))
        else:
            return None
        
    def remplacement_val_manquantes2(self, method, attribut):
        missing = [i for i, val in enumerate(self.dataset[:, attribut]) if np.isnan(val)]
        
        for i in missing:
            zone = self.dataset[i, 0]
            time_period = self.dataset[i, 1]
            matching_rows = [z for z in range(self.dataset.shape[0]) if self.dataset[z, 1] == time_period and not np.isnan(self.dataset[z, attribut])]
            if method == "Mode":
                if matching_rows:
                    mode = statistics.mode(self.dataset[matching_rows, attribut])
                    self.dataset[i, attribut] = mode
                else:
                    zone_rows = [z for z in range(self.dataset.shape[0]) if self.dataset[z, 0] == zone and not np.isnan(self.dataset[z, attribut])]
                    mode = statistics.mode(self.dataset[zone_rows, attribut])
                    self.dataset[i, attribut] = mode
            else:
                if matching_rows:
                    mean_val = np.mean(self.dataset[matching_rows, attribut])
                    self.dataset[i, attribut] = mean_val
                else:
                    zone_rows = [z for z in range(self.dataset.shape[0]) if self.dataset[z, 0] == zone and not np.isnan(self.dataset[z, attribut])]
                    mean_val = np.mean(self.dataset[zone_rows, attribut])
                    self.dataset[i, attribut] = mean_val

    def remplacement_manquant_generale2(self, method):
        for attribute_index in self.numeric_columns:
           self.remplacement_val_manquantes2(method, attribute_index)
        
    def remplacement_aberantes_generale2(self, method):
        if method=="Discritisation":
            categorical_columns = [0, 1, 2]
            self.numeric_columns = [col for col in self.numeric_columns if col not in categorical_columns]
        for attribute_index in self.numeric_columns:
            self.remplacement_val_aberrantes(method, attribute_index)

class AttributeAnalyzer:
    def __init__(self, dataset, dataFrame):
        self.dataFrame = dataFrame
        self.dataset = dataset
        
    def val_manquante(self, attribute):
        L=[]
        for i in range(0,len(self.dataset[:,attribute])):
            if not re.fullmatch(r"\d+\.(:?\d+)?", str(self.dataset[i, attribute])):
                L.append(i)
        return L
    
    def calcul_mediane(self, attribute):
        datasetCurrated=np.delete(self.dataset[:,attribute], self.val_manquante(attribute))
        liste = cp.deepcopy(datasetCurrated)
        liste.sort()
        if liste.size % 2 !=0 :
        
            mediane=liste[((liste.size+1)//2) -1]
        else :
            mediane=(liste[(liste.size//2)-1]+liste[liste.size//2])/2
        return mediane

    def tendance_centrales_homeMade(self, attribute):
        datasetCurrated=np.delete(self.dataset[:,attribute], self.val_manquante(attribute))
        moyenne2 = datasetCurrated.sum() / datasetCurrated.shape[0]
        mediane2 = self.calcul_mediane(attribute)
        unique_values, counts = np.unique(datasetCurrated, return_counts=True)
        Indicemax = np.where(counts == max(counts))[0]
        mode2=[unique_values[i] for i in Indicemax]
        return [moyenne2,mediane2,mode2]
    
    def quartilles_homeMade(self, attribute):
        datasetCurrated=np.delete(self.dataset[:,attribute], self.val_manquante(attribute))
        liste = cp.deepcopy(datasetCurrated)
        liste.sort()
        q0=liste[0]
        q1=(liste[liste.size//4-1]+liste[liste.size//4]) /2
        q3=(liste[liste.size*3//4-1]+liste[liste.size*3//4]) /2
        q2=self.calcul_mediane(attribute)
        q4=liste[-1]
        return [q0,q1,q2,q3,q4]
    
    def ecart_type_home_made(self, attribute):
        datasetCurrated=np.delete(self.dataset[:,attribute], self.val_manquante(attribute))
        mean = np.mean(datasetCurrated)
        ecarts = [(val - mean) ** 2 for val in datasetCurrated]
        variance = np.mean(ecarts) 
        return np.sqrt(variance)
    
    def Boite_a_moustache(self, attribute,boolen):
        abberante=[]
        liste=[]
        q3=self.quartilles_homeMade(attribute)[-2]
        q1=self.quartilles_homeMade(attribute)[1]
        IQR=(self.quartilles_homeMade(attribute)[-2]-self.quartilles_homeMade(attribute)[1])*1.5
        datasetCurrated=np.delete(self.dataset[:,attribute], self.val_manquante(attribute))

        for var in datasetCurrated:
            if (var <(q3+IQR) and var>(q1-IQR)):
                liste.append(var)
            else:
                abberante.append(var)  
        if boolen:
            plt.boxplot(datasetCurrated)
        
        else:
            plt.boxplot(liste)

    def scatterplot(self, attribute, attribute2):
        plt.scatter(self.dataset[:,attribute],self.dataset[:,attribute2],marker ='p')
        plt.title(f'Scatter Plot of the attributes {self.dataFrame.columns.tolist()[attribute]} and {self.dataFrame.columns.tolist()[attribute2]}')
        plt.xlabel(f'{self.dataFrame.columns.tolist()[attribute]} Attribute values')
        plt.ylabel(f'{self.dataFrame.columns.tolist()[attribute2]} Attribute values')
   

    def histogramme(self, attribute):
        datasetCurrated=np.delete(self.dataset[:,attribute], self.val_manquante(attribute))
        plt.hist(datasetCurrated, bins=math.ceil(1+(10/3)*np.log10(self.dataset.shape[0])),edgecolor='black')
        plt.title(f'Histograme of the attribute {self.dataFrame.columns.tolist()[attribute]}')
        plt.xlabel('Attribute values')
        plt.ylabel('Frequence')
    
    def attribute_infos(self, attribute, outliers, scatter_attribute):
        moyenne2, mediane2, mode2 = self.tendance_centrales_homeMade(attribute)
        q0, q1, q2, q3, q4 = self.quartilles_homeMade(attribute)
        ecart_type = self.ecart_type_home_made(attribute)

        hist_fig = plt.figure()
        self.histogramme(attribute)
        hist_fig.savefig("histogramme.png")
        plt.close(hist_fig)

        box_plot_fig = plt.figure()
        self.Boite_a_moustache(attribute, outliers)  
        box_plot_fig.savefig("boxPlot.png")
        plt.close(box_plot_fig)

        scatter_plot_fig = plt.figure()
        self.scatterplot(attribute, scatter_attribute)
        scatter_plot_fig.savefig("scatterPlot.png")
        plt.close(scatter_plot_fig)

        plots = ["histogramme.png", "boxPlot.png", "scatterPlot.png"]

        return moyenne2, mediane2, mode2, q0, q1, q2, q3, q4, ecart_type, plots

class StatisticsCOVID19:
    def __init__(self, df):
        self.df = pd.DataFrame(df)
        self.df['Year'] = pd.to_datetime(self.df['Start date']).dt.year
        self.df['Month'] = pd.to_datetime(self.df['Start date']).dt.month

    def plot_total_cases_and_positive_tests(self, chosen_attribute):
        totals = self.df.groupby('zcta')[[chosen_attribute]].sum().reset_index()
        bar_width = 0.5
        index = totals.index
        plt.bar(index, totals[chosen_attribute], bar_width, label=chosen_attribute)
        plt.xlabel('Zones')
        plt.ylabel('Count')
        plt.title(f'Distribution du nombre total de {chosen_attribute} par zones')

    def plot_total_cases_and_positive_tests_treemap(self, chosen_attribute):
        totals = self.df.groupby('zcta')[[chosen_attribute]].sum().reset_index()

        totals['value_normalized'] = totals[chosen_attribute] / totals[chosen_attribute].sum()

        totals = totals.sort_values(by=chosen_attribute, ascending=False)

        colors = plt.cm.tab10(range(len(totals)))

        fig, ax = plt.subplots(figsize=(10, 6))
        squarify.plot(
            sizes=totals['value_normalized'],
            label=totals['zcta'],
            color=colors,
            alpha=0.7,
            ax=ax 
        )

        plt.title(f'Distribution du nombre de {chosen_attribute} par zone')
        plt.axis('off') 
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)

        # Save the buffer to the specified filename
        with open("total_cases_and_positive_tests_treemap_plot.png", 'wb') as f:
            f.write(buffer.getvalue())

        return "total_cases_and_positive_tests_treemap_plot.png" 

    def weekly_plot(self, chosen_zone, chosen_year, chosen_month, chosen_attribute):

        zone_df = self.df[self.df['zcta'] == chosen_zone]

        hebdo_df = zone_df[(zone_df['Month'] == chosen_month) & (zone_df['Year'] == chosen_year)]

        plt.plot(hebdo_df['Start date'], hebdo_df[chosen_attribute], label=chosen_attribute)
        plt.title(f'L\'évolution hebdomadaire du total de {chosen_attribute} pour la zone {chosen_zone} pendant le {chosen_month} ème mois de l\'année {chosen_year}')
        plt.xlabel('Dates')
        plt.ylabel('Count')

    def monthly_plot(self, chosen_zone, chosen_year, chosen_attribute):

        zone_df = self.df[self.df['zcta'] == chosen_zone]

        monthly_df = zone_df[zone_df['Year'] == chosen_year]
        month_df = monthly_df.groupby('Month')[[chosen_attribute]].sum().reset_index()
        plt.plot(month_df['Month'], month_df[chosen_attribute], label=chosen_attribute)
        plt.title(f'L\'évolution mensuelle du total de {chosen_attribute} pour la zone {chosen_zone} pendant l\'année {chosen_year}')
        plt.xlabel('Months')
        plt.ylabel('Count') 
        
    def annual_plot(self, chosen_zone, chosen_attribute):

        zone_df = self.df[self.df['zcta'] == chosen_zone]

        annual_df = zone_df.groupby('Year')[[chosen_attribute]].sum().reset_index()

        plt.plot(annual_df['Year'], annual_df[chosen_attribute], label=chosen_attribute)
        plt.title(f'L\'évolution annuelle du total de {chosen_attribute} pour la zone {chosen_zone}')
        plt.xlabel('Years')
        plt.ylabel('Count') 
        plt.xticks(annual_df['Year'])


    def stacked_bar_plot(self):
        self.df = self.df.sort_values(by=['Year'])  # Sort the DataFrame by 'Year'

        grouped_data = self.df.groupby(['Year', 'zcta'])['case count'].sum().unstack()

        years = self.df['Year'].unique()
        zone_columns = grouped_data.columns

        bottom_values = None

        for zone in zone_columns:
            values = grouped_data[zone].reindex(years, fill_value=0).values
            if bottom_values is None:
                plt.bar(years, values, label=zone)
                bottom_values = values
            else:
                plt.bar(years, values, label=zone, bottom=bottom_values)
                bottom_values += values

        plt.title('Stacked Bar Graph of Case Count by Zone and Year')
        plt.xlabel('Year')
        plt.ylabel('Case Count')
        plt.legend(title='Zone', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(years) 


    def pop_tests_plot(self):
        data = self.df[['population', 'test count']]
        grouped_data = data.groupby('population').sum().reset_index()

        plt.plot(grouped_data['population'], grouped_data['test count'])
        plt.title('Population vs Test Count')
        plt.xlabel('Population')
        plt.ylabel('Test count')

    def plot_top_zones_impacted(self, n):
        grouped_data = self.df.groupby('zcta')['case count'].sum().sort_values(ascending=True).head(n)
        bar_width = 0.1

        grouped_data.plot.barh(figsize=(12, 6), width=bar_width)
        plt.title(f'Top {n} Zones les plus impactées par le Coronavirus')
        plt.xlabel('Nombre de Cas')

    def plot_time_period_data(self, chosen_time_period, chosen_attribute):
        selected_data = self.df[(self.df['time_period'] == chosen_time_period)]
        grouped_data = selected_data.groupby('zcta')[[chosen_attribute]].sum().reset_index()

        bar_width = 0.2
        index = grouped_data.index

        plt.bar(index - bar_width, grouped_data[chosen_attribute], width=bar_width, label=chosen_attribute)
        plt.xticks(index - bar_width, grouped_data['zcta'])
        plt.xlabel('Zone (zcta)')
        plt.ylabel('Count')
        plt.title(f'Total of {chosen_attribute} for Time Period {chosen_time_period}')
        plt.legend()

class FrequentItemsets:
    def Ck_generator(self, k, L, datasetT):
        if k==1:
            uniItemListe=set()
            for i in range(0,len(datasetT)):
                uniItemListe.update(set(datasetT[i]))
            return [(v,) for v in uniItemListe]
        else:
            C=[]
            if len(L)==0: return C
            listeitemsunique=sorted(set([element for tuple in L.keys() for element in tuple]))
            if len(listeitemsunique)<k: return []
            combinations_list = list(combinations(listeitemsunique, k))
            if k==2: 
                return combinations_list
            #le pruning
            for combi in combinations_list:
                exist=True
                sous_combinations_list = list(combinations(combi, k-1))
                
                for sous_combi in sous_combinations_list:
                    if sous_combi not in list(L.keys()):
                        exist=False
                        break
                if exist==True:
                    C.append(combi)   
            return C
        
    def support_calculator(self, C, datasetT):
        dico={}
        dico.update({val:0 for val in C})
        for row in datasetT:
            combinations_list = list(combinations(row, len(list(dico.keys())[0])))
            for val in combinations_list:
                if val in dico: 
                    dico[val]+=1
        dico.update({key:val/len(datasetT) for key,val in dico.items()})
    
        return dico

    def Lk_generator(self, C, supp_min):
        c={}
        c=({key:float(val) for key,val in C.items() if float(val)>=supp_min})
        return c
    
    def appriori(self, min_supp, datasetT):
        L=[]
        k=1
        C=self.Ck_generator(k,None,datasetT)
    
        while(len(C)!=0):
            S=self.support_calculator(C,datasetT)
        
            l=self.Lk_generator(S,min_supp)
            if len(l)!=0:L.append(l)
        
            k+=1
            C=self.Ck_generator(k,l,None)
            
        return L
    
    def regle_association(self, L):
        regles=pd.DataFrame()
        first=True
        for key,value in L.items():
            if first==True:
                k=len(key)
                first==False
            if k==2 :
                    new_row = {'antecedant':(key[0],),'consequent':(key[1],),'mesure':0.0,'support':value}
                    regles = pd.concat([regles, pd.DataFrame([new_row])], ignore_index=True)
                    new_row={'antecedant':(key[1],),'consequent':(key[0],),'mesure':0.0,'support':value}
                    regles = pd.concat([regles, pd.DataFrame([new_row])], ignore_index=True)
            else:
                for i in range(k-1,k-(k//2)-1,-1):
                    sous_combinations_list = list(combinations(key,i))#ab c  / ac bc 
                    for sous_comb in sous_combinations_list:
                        reste=sorted(set(key).symmetric_difference(sous_comb))
                        regles = pd.concat([regles, pd.DataFrame([{'antecedant':tuple(reste),'consequent':sous_comb,'mesure':0.0,'support':value}])], ignore_index=True)
                        if not len(tuple(reste))==len(sous_comb):
                            regles = pd.concat([regles, pd.DataFrame([{'antecedant':sous_comb,'consequent':tuple(reste),'mesure':0.0,'support':value}])], ignore_index=True)
            
        return regles
    
    def mesure_calculator(self, r, methode, L):#A,D->B,C
        if methode==0:#confidence
            return (r["support"]/L[len(r["antecedant"])-1][r["antecedant"]])
        elif methode==1:#cosine
            return (r["support"]/math.sqrt(L[len(r["antecedant"])-1][r["antecedant"]]*L[len(r["consequent"])-1][r["consequent"]]))
        elif methode==2:#lift
            return r["support"]/(L[len(r["antecedant"])-1][r["antecedant"]]*L[len(r["consequent"])-1][r["consequent"]])
        elif methode==3:#jackard
            return r["support"]/(L[len(r["antecedant"])-1][r["antecedant"]]+L[len(r["consequent"])-1][r["consequent"]]- r["support"])
        else:#kulc
            return 0.5*((r["support"]/L[len(r["antecedant"])-1][r["antecedant"]])+(r["support"]/L[len(r["consequent"])-1][r["consequent"]]))

    def regles_frequente(self, L, conf_min, m):

        regles=pd.DataFrame()
        for l in L[1:]:
            regles= pd.concat([regles, self.regle_association(l)], ignore_index=True)
            

        for i in range(0,len(regles)):
            regles.iloc[i,2] = self.mesure_calculator(regles.loc[i],m,L)
        
        if not len(regles)==0:
            regles=regles[regles['mesure'] >= conf_min]
            
        return regles
    
    def rules_nbr_plot(self, transactions_table, supp_lower_bound, supp_upper_bound, conf_lower_bound, conf_upper_bound):
        results= np.empty((0,3),float)
        for sup_min in np.arange(supp_lower_bound,supp_upper_bound,0.01):
            for conf_min in np.arange(conf_lower_bound,conf_upper_bound,0.01):
                L=self.appriori(sup_min, transactions_table)
                rs=self.regles_frequente(L,conf_min,0)
            
                results=np.vstack((results,np.array([sup_min,conf_min,len(rs)])))

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

        Z = results[:, 2].reshape(results.shape[0], 1)
        ax.scatter(results[:, 0], results[:, 1], Z, c='r', marker='o')

        tri = Delaunay(results[:, :2])

        ax.plot_trisurf(results[:, 0], results[:, 1], Z.flatten(), triangles=tri.simplices, cmap="viridis", linewidth=0.9, antialiased=True)

        ax.set_xlabel('sup')
        ax.set_ylabel('conf')
        ax.set_zlabel('nbr de regles frequentes', labelpad=10)

        ax.view_init(elev=10, azim=-40)
        plt.title("Nombre de regles frequentes générées par sup_min et conf_min")
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)

        with open("rules_nbr_plot.png", 'wb') as f:
            f.write(buffer.getvalue())

        return "rules_nbr_plot.png"
        
    def freq_items_nbr_plot(self, transactions_table, supp_lower_bound, supp_upper_bound):
        resultsf= np.empty((0,2),float)
        for sup_min in np.arange(supp_upper_bound,supp_lower_bound,-0.001):
            L=self.appriori(sup_min,transactions_table)
            resultsf=np.vstack((resultsf,np.array([sup_min,sum(len(l) for l in L)]))) 
        
        x = resultsf[:,0]
        y = resultsf[:,1]

        plt.plot(x, y)
        plt.title('Evolution du nombre de motifs frequents selon le support min')
        plt.xlabel('supmin')
        plt.ylabel('nbr motifs frequents')

    def time_exec_plot(self, transactions_table, supp_lower_bound, supp_upper_bound):
        TimeResults= np.empty((0,2),float)

        for sup_min in np.arange(supp_upper_bound,supp_lower_bound,-0.002):
            duree=0.0
            for j in range(0,10):
                start=time.time()
                L=self.appriori(sup_min,transactions_table)
                duree+=time.time()-start
            TimeResults=np.vstack((TimeResults,np.array([sup_min,duree/10.0])))
                    
        x = TimeResults[:,0]
        y = TimeResults[:,1]

        plt.plot(x, y)
        plt.title('Evolution du temps d''execution d''apriori selon le sup_min')
        plt.xlabel('Supmin')
        plt.ylabel('Temps d execution')

    def memory_alloc_plot(self, transactions_table, supp_lower_bound, supp_upper_bound, conf_lower_bound, conf_upper_bound):
        resultsMemory= np.empty((0,3),float)

        for sup_min in np.arange(supp_lower_bound,supp_upper_bound,0.06):
            for conf_min in np.arange(conf_lower_bound,conf_upper_bound,0.02):
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024 /2024 # in MB
                L=self.appriori(sup_min,transactions_table)
                rs=self.regles_frequente(L,conf_min,0)
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024 /2024 # in MB
                resultsMemory=np.vstack((resultsMemory,np.array([sup_min,conf_min,final_memory-initial_memory])))

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

        Z = resultsMemory[:, 2].reshape(resultsMemory.shape[0], 1)
        ax.scatter(resultsMemory[:, 0], resultsMemory[:, 1], Z, c='r', marker='o')

        # Create Delaunay triangulation
        tri = Delaunay(resultsMemory[:, :2])

        # Plot the surface using the triangulation
        ax.plot_trisurf(resultsMemory[:, 0], resultsMemory[:, 1], Z.flatten(), triangles=tri.simplices, cmap="viridis", linewidth=0.9, antialiased=True)

        ax.set_xlabel('sup__min')
        ax.set_ylabel('conf_min')
        ax.set_zlabel('memoire',labelpad=10)
        ax.view_init(elev=10, azim=-30)
        plt.title("Evolution de l'espace aloué à l''algorithme apriori et regles d''association selon le sup_min et conf_min")
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)

        # Save the buffer to the specified filename
        with open("memory_alloc_plot.png", 'wb') as f:
            f.write(buffer.getvalue())

        return "memory_alloc_plot.png"


class App:
    def __init__(self):
        self.dataFrame1 = pd.read_csv('Dataset1.csv')
        self.dataset1 = np.genfromtxt('Dataset1.csv', delimiter=',', dtype=float, skip_header=1)
        self.attribute_analyzer = AttributeAnalyzer(self.dataset1, self.dataFrame1)
        self.preprocessor1 = Preprocessing(self.dataset1, self.dataFrame1)
        self.dataFrame2 = pd.read_csv('Dataset2.csv')
        self.dataFrame2 = self.dataFrame2.replace({pd.NA: np.nan})
        self.dataset2 = self.dataFrame2.to_numpy()
        self.preprocessor2 = Preprocessing(self.dataset2, self.dataFrame2)
        self.dataFrame3 = pd.read_csv('Dataset3.xlsx - 8.forFMI.csv', delimiter=',', decimal=',')
        self.dataset3 = self.dataFrame3.to_numpy()
        self.FIL = FrequentItemsets()
        self.selected_attribute_dataset3 = 0
        self.create_interface()
        
    def infos_dataset(self, dataFrame):
        num_rows, num_cols = pd.DataFrame(dataFrame).shape
        attr_desc = pd.DataFrame(dataFrame).describe()
        attr_desc.insert(0, 'Stats', attr_desc.index)
        return num_rows, num_cols, attr_desc
    
    def preprocessing_general2(self, manque_meth, aberrante_meth):
        for row in self.dataset2:
            row[3] = self.preprocessor2.convert_date(row[1], row[3])
            row[4] = self.preprocessor2.convert_date(row[1], row[4])
        self.preprocessor2.remplacement_manquant_generale2(manque_meth)
        self.preprocessor2.remplacement_aberantes_generale2(aberrante_meth)
        self.dataset2 = self.preprocessor2.dataset
        return self.dataset2
    
    def plots(self, df, graph, graph_type1, attribute1, zone2, attribute2, period2, year2, month2, year22, n5, time_period6, attribute6):
        stats = StatisticsCOVID19(df)
        if graph == "Total des cas confirmés et tests positifs par zones":
            if graph_type1 == "Bar Chart":
                plot1 = plt.figure()
                stats.plot_total_cases_and_positive_tests(attribute1)
                plot1.savefig("plot1.png")
                plt.close(plot1)
                plot = ["plot1.png"]
                return plot
            
            if graph_type1 == "Tree Map":
                plot1 = stats.plot_total_cases_and_positive_tests_treemap(attribute1)
                plot = [plot1]
                return plot
            
        if graph == "Evolution du virus au fil du temps":
            if period2 == "Weekly":
                plot2 = plt.figure()
                stats.weekly_plot(zone2, year2, month2, attribute2)
                plot2.savefig("plot2_weekly.png")
                plt.close(plot2)
                plot = ["plot2_weekly.png"]
                return plot
            
            if period2 == "Monthly":
                plot2 = plt.figure()
                stats.monthly_plot(zone2, year22, attribute2)
                plot2.savefig("plot2_monthly.png")
                plt.close(plot2)
                plot = ["plot2_monthly.png"]
                return plot
            
            if period2 == "Annual":
                plot2 = plt.figure()
                stats.annual_plot(zone2, attribute2)
                plot2.savefig("plot2_annual.png")
                plt.close(plot2)
                plot = ["plot2_annual.png"]
                return plot
            
        if graph == "Total des cas par zone et par année":
            plot3 = plt.figure()
            stats.stacked_bar_plot()
            plot3.savefig("plot3.png")
            plt.close(plot3)
            plot = ["plot3.png"]
            return plot
        
        if graph == "Rapport entre la population et le nombre de tests effectués":
            plot4 = plt.figure()
            stats.pop_tests_plot()
            plot4.savefig("plot4.png")
            plt.close(plot4)
            plot = ["plot4.png"]
            return plot
        
        if graph == "Top 5 des zones les plus impactées par le coronavirus":
            plot5 = plt.figure()
            stats.plot_top_zones_impacted(n5)
            plot5.savefig("plot5.png")
            plt.close(plot5)
            plot = ["plot5.png"]
            return plot
        
        if graph == "Rapport entre les cas confirmés, les tests effectués et les tests positifs a une periode choisie pour chaque zone":
            plot6 = plt.figure()
            stats.plot_time_period_data(time_period6, attribute6)
            plot6.savefig("plot6.png")
            plt.close(plot6)
            plot = ["plot6.png"]
            return plot

    def Discretisation(self, method, attribute, K):
        vals=self.dataset3[:,attribute].copy()
        vals.sort()

        if method == "Equal-Width Discretization":
            #Equal-Width Discretization
            nbrelmt=math.ceil(self.dataset3[:,attribute].shape[0]/K)
            
            for val in range(0,self.dataset3[:,attribute].shape[0]):  
                for i in range(0,vals.shape[0],nbrelmt):
                    if(vals[i]>self.dataset3[val,attribute]):
                        sup=i
                        break
                self.dataset3[val,attribute]=np.mean(vals[sup-nbrelmt:sup])       
        else:
            #Equal-Frequency Discretization
            largeur= (self.dataset3[:,attribute].max() - self.dataset3[:,attribute].min())/math.ceil(K)
            
            dic={}
            bornesup= self.dataset3[:,attribute].min()+largeur
            for val in vals:
                if val>=bornesup and bornesup<self.dataset3[:,attribute].max():
                    bornesup+=largeur

                if bornesup in dic:   
                    dic[bornesup].append(val)
                else:
                    dic[bornesup]=[val]

            for i in range(0,self.dataset3[:,attribute].shape[0]):
                for item in dic.items():
                    if (self.dataset3[i,attribute]>=item[0]-largeur and self.dataset3[i,attribute]<item[0]):
                        self.dataset3[i,attribute]=np.mean(item[1])
                        break

    def discretization_plot(self, method, attribute, K):
        plot = plt.figure()
        #Before Discretization
        plt.hist(self.dataset3[:,attribute], bins=K, edgecolor='black')
        
        #Discretization
        self.Discretisation(method, attribute, K)
        self.transactions_table = np.array([np.sort(list(four)) for four in zip(self.dataset3[:, attribute].astype(str), self.dataset3[:,3], self.dataset3[:,4], self.dataset3[:,5])])

        #After Discretization
        attribute_label = self.dataFrame3.columns[attribute]
        freq=Counter(self.dataset3[:, attribute])
        x_values = np.sort(list(freq.keys()))
        y_values = [freq[x] for x in x_values]
        
        plt.plot(x_values, y_values)
        plt.xlabel(f'{attribute_label}')
        plt.ylabel('Frequency')
        plt.title(f'Discretized {attribute_label}')
        plot.savefig("discretization_plot.png")
        plt.close(plot)
        plot = ["discretization_plot.png"]
        return self.dataset3, plot, self.transactions_table

    def FIL_general(self, transactions_table, supp_min, conf_min, metric):
        self.transactions_table = pd.DataFrame(transactions_table).to_numpy()
        self.L = self.FIL.appriori(supp_min, self.transactions_table)
        self.association_rules = self.FIL.regles_frequente(self.L, conf_min, metric)
        self.L = [(k, v) for dictionary in self.L for k, v in dictionary.items()]

        return self.association_rules, pd.DataFrame(self.L, columns=['Frequent Itemset', 'Support'])

    def experimentation_plots(self, supp_lower_bound, supp_upper_bound, conf_lower_bound, conf_upper_bound):
        
        plot1 = self.FIL.rules_nbr_plot(self.transactions_table, supp_lower_bound, supp_upper_bound, conf_lower_bound, conf_upper_bound)

        plot2 = plt.figure()
        self.FIL.freq_items_nbr_plot(self.transactions_table, supp_lower_bound, supp_upper_bound)
        plot2.savefig("freq_items_nbr_plot.png")
        plt.close(plot2)

        plot3 = plt.figure()
        self.FIL.time_exec_plot(self.transactions_table, supp_lower_bound, supp_upper_bound)
        plot3.savefig("time_exec_plot.png")
        plt.close(plot3)

        plot4 = self.FIL.memory_alloc_plot(self.transactions_table, supp_lower_bound, supp_upper_bound, conf_lower_bound, conf_upper_bound)

        plots = [plot1, "freq_items_nbr_plot.png", "time_exec_plot.png", plot4]
        
        return plots

    def recommendation(self, a1, a2, a3, method):
        print("##############################", a1, a2, a3)
        print("**************************************",method)
        instance = []
        if a1 != None:
            instance.append(a1)
        if a2 != None:
            instance.append(a2)
        if a3 != None:
            instance.append(a3)

        r_filtered=[]
        for index, row in self.association_rules.iterrows():
            if method == "Strict":
                if row[0]==tuple(sorted(instance)):
                    r_filtered.append(row)
            else:
                if set(list(row[0])).issubset(set(instance)):
                    r_filtered.append(row)

        if r_filtered:
            max_consequent_length = max(len(r["consequent"]) for r in r_filtered)
        else:
            max_consequent_length = 0

        consequent_columns = [f"Consequent {i+1}" for i in range(max_consequent_length)]

        return pd.DataFrame([r["consequent"] for r in r_filtered], columns=consequent_columns)

    def create_interface(self):
        with gr.Blocks() as demo:
            with gr.Tab("Agriculture"):
                with gr.Tab("Dataset Visualization"):
                    with gr.Column():
                        gr.Markdown("""# Dataset1 Analysis Dashboard""")
                        
                        with gr.Row():
                            inputs = [gr.Dataframe(label="Dataset1", value=self.dataFrame1)]
                           
                            with gr.Column():
                                outputs = [gr.Textbox(label="Number of Rows"), gr.Textbox(label="Number of Columns"), gr.Dataframe(label="Attributes description")]
                        
                        btn = gr.Button("Submit")
                        btn.click(fn=self.infos_dataset, inputs=inputs, outputs=outputs)

                with gr.Tab("Attributes"):
                    with gr.Column():
                        gr.Markdown("""# Attributes Analysis""")
                        
                        with gr.Row():
                            inputs = [gr.Dropdown([(f"{att}", i) for i, att in enumerate(self.dataFrame1.columns.tolist())], multiselect=False, label="Attributes", info="Select an attribute : "), gr.Radio(["With Outliers", "Without Outliers"], label="Box Plot Parameters"), gr.Dropdown([(f"{att}", i) for i, att in enumerate(self.dataFrame1.columns.tolist())], multiselect=False, label="Scatter Plot Parameters", info="Select a second attribute for the scatter plot : ")]
                        
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    trends = [gr.Textbox(label="Mean"), gr.Textbox(label="Median"), gr.Textbox(label="Mode")]
                                with gr.Row():
                                    quartiles = [gr.Textbox(label="Q0"), gr.Textbox(label="Q1"), gr.Textbox(label="Q2"), gr.Textbox(label="Q3"), gr.Textbox(label="Q4")]
                                with gr.Row():
                                    deviation = [gr.Textbox(label="Standard Deviation")]

                            with gr.Column():
                                gallery = [gr.Gallery(label="Attribute Visualisation", columns=(1,2))]
                            
                        with gr.Row():
                            gr.ClearButton(inputs)
                            btn = gr.Button("Submit")
                            btn.click(fn=self.attribute_analyzer.attribute_infos, inputs=inputs, outputs=trends+quartiles+deviation+gallery)
            
                with gr.Tab("Preprocessing"):
                    with gr.Column():
                        gr.Markdown("""# Preprocessing of Dataset1""")

                        with gr.Row():
                            inputs = [gr.Dropdown(["Mode", "Mean"], multiselect=False, label="Missing Values", info="Select a method to handle the missing values in the dataset :"), 
                                gr.Dropdown(["Linear Regression", "Discritisation"], multiselect=False, label="Outliers", info="Select a method to handle the outliers in the dataset :"), 
                                gr.Dropdown(["Vmin-Vmax", "Z-Score"], multiselect=False, label="Normalization", info="Select a method to normalize the dataset :"),
                                gr.Textbox(label="Vmin", visible=True, interactive=True, value=0),
                                gr.Textbox(label="Vmax", visible=True, interactive=True, value=0)]
                            
                        with gr.Row():
                            outputs = [gr.Dataframe(label="Dataset1 preprocessed", headers=self.dataFrame1.columns.tolist())]

                        with gr.Row():
                            gr.ClearButton(inputs)
                            btn = gr.Button("Submit")
                            btn.click(fn=self.preprocessor1.preprocessing_general1, inputs=inputs, outputs=outputs)

                with gr.Tab("Classification"):
                    gr.Markdown("hh")
                with gr.Tab("Clustering"):
                    gr.Markdown("hh")
                with gr.Tab("Recommender"):
                    gr.Markdown("hh")                   

            with gr.Tab("COVID-19"):
                with gr.Tab("Dataset Visualization"):
                    with gr.Column():
                        gr.Markdown("""# Dataset2 Analysis Dashboard""")
                        
                        with gr.Row():
                            inputs = [gr.Dataframe(label="Dataset2", value=self.dataFrame2)]
                           
                            with gr.Column():
                                outputs = [gr.Textbox(label="Number of Rows"), gr.Textbox(label="Number of Columns"), gr.Dataframe(label="Attributes description")]
                        
                        btn = gr.Button("Submit")
                        btn.click(fn=self.infos_dataset, inputs=inputs, outputs=outputs)

                with gr.Tab("Preprocessing"):
                    with gr.Column():
                        gr.Markdown("""# Preprocessing of Dataset2""")

                        with gr.Row():
                            inputs = [gr.Dropdown(["Mode", "Mean"], multiselect=False, label="Missing Values", info="Select a method to handle the missing values in the dataset :"), 
                                gr.Dropdown(["Linear Regression", "Discritisation"], multiselect=False, label="Outliers", info="Select a method to handle the outliers in the dataset :")]
                            
                        with gr.Row():
                            outputs_preprocess = gr.Dataframe(label="Dataset2 preprocessed", headers=self.dataFrame2.columns.tolist())

                        with gr.Row():
                            gr.ClearButton(inputs)
                            btn = gr.Button("Submit")
                            btn.click(fn=self.preprocessing_general2, inputs=inputs, outputs=outputs_preprocess)

                with gr.Tab("Statistics"):
                    with gr.Column():
                        with gr.Row():
                            graph = gr.Dropdown(["Total des cas confirmés et tests positifs par zones", "Evolution du virus au fil du temps", "Total des cas par zone et par année",
                                                 "Rapport entre la population et le nombre de tests effectués", 
                                                 "Top 5 des zones les plus impactées par le coronavirus",
                                                 "Rapport entre les cas confirmés, les tests effectués et les tests positifs a une periode choisie pour chaque zone"], multiselect=False, label="Graphs", info="Select a graph to plot :")
                            with gr.Row(visible=False) as row:
                                plot1_param = [gr.Radio(["Tree Map", "Bar Chart"]),
                                               gr.Dropdown(['case count', 'positive tests'], multiselect=False, label="Atribute", info="Choose an attribute to plot :")]

                            with gr.Row(visible=False) as row2:
                                plot2_param = [gr.Dropdown(self.dataFrame2['zcta'].unique().tolist(), multiselect=False, label="Zone", info="Select a zone to plot :"), 
                                               gr.Dropdown(["case count", "test count", "positive tests"], multiselect=False, label="Attribute", info="Select an attribute to plot :"), 
                                               gr.Dropdown(["Weekly", "Monthly", "Annual"], multiselect=False, label="Period", info="Select a period to plot :")]
                                with gr.Row(visible=False) as row2_w:
                                    weekly_param = [gr.Dropdown([2019, 2020, 2021, 2022], multiselect=False, label="Year", info="Choose a year:"),
                                                    gr.Dropdown([i+1 for i in range(12)], multiselect=False, label="Month", info="Choose a month:")]
                                with gr.Row(visible=False) as row2_m:
                                    monthly_param = [gr.Dropdown([2019, 2020, 2021, 2022], multiselect=False, label="Year", info="Choose a year:")]
                                
                            with gr.Row(visible=False) as row5:
                                plot5_param = [gr.Slider(1, 7, value=5, label="Number of zones", info="Choose between 1 and 7 zones", step=1)]

                            with gr.Row(visible=False) as row6:
                                plot6_param = [gr.Dropdown(np.unique(self.dataset2[:, 1]).tolist(), multiselect=False, label="time period", info="Select a time period to plot :"),
                                    gr.Dropdown(["case count", "test count", "positive tests"], multiselect=False, label="Attribute", info="Select an attribute to plot :")]


                                def update_visibility(selected_graph):
                                    if selected_graph == "Total des cas confirmés et tests positifs par zones":
                                        return {row: gr.Row(visible=True),
                                                row2: gr.Row(visible=False), 
                                                row5: gr.Row(visible=False), 
                                                row6: gr.Row(visible=False)}
                                    if selected_graph == "Evolution du virus au fil du temps":
                                        return {row2: gr.Row(visible=True),
                                                row: gr.Row(visible=False), 
                                                row5: gr.Row(visible=False), 
                                                row6: gr.Row(visible=False)}
                                    if selected_graph == "Total des cas par zone et par année" or selected_graph == "Rapport entre la population et le nombre de tests effectués":
                                        return {row2: gr.Row(visible=False),
                                                row: gr.Row(visible=False),
                                                row5: gr.Row(visible=False), 
                                                row6: gr.Row(visible=False)}
                                    if selected_graph == "Top 5 des zones les plus impactées par le coronavirus":
                                        return {row2: gr.Row(visible=False),
                                                row: gr.Row(visible=False),
                                                row5: gr.Row(visible=True), 
                                                row6: gr.Row(visible=False)}
                                    if selected_graph == "Rapport entre les cas confirmés, les tests effectués et les tests positifs a une periode choisie pour chaque zone":
                                        return {row2: gr.Row(visible=False),
                                                row: gr.Row(visible=False),
                                                row5: gr.Row(visible=False), 
                                                row6: gr.Row(visible=True)}
                                    
                                def update_visibility_param(selected_period):
                                    if selected_period == "Weekly":
                                        return {row2_w: gr.Row(visible=True),
                                                row2_m: gr.Row(visible=False)}
                                    if selected_period == "Monthly":
                                        return {row2_w: gr.Row(visible=False),
                                                row2_m: gr.Row(visible=True)}
                                    else:
                                        return {row2_w: gr.Row(visible=False),
                                                row2_m: gr.Row(visible=False)}
                                    
                                graph.change(update_visibility, inputs=graph, outputs=[row, row2, row5, row6])
                                plot2_param[2].change(update_visibility_param, inputs=plot2_param[2], outputs=[row2_w, row2_m])
                        

                        with gr.Row():
                            outputs = [gr.Gallery(label="Graphs", columns=(1,2))]
                        
                        with gr.Row():
                            gr.ClearButton(inputs)
                            btn = gr.Button("Submit")
                            btn.click(fn=self.plots, inputs=[outputs_preprocess]+[graph]+plot1_param+plot2_param+weekly_param+monthly_param+plot5_param+plot6_param, outputs=outputs)
            
            with gr.Tab("Frequent Itemset Learning"):
                with gr.Tab("Dataset Visualization"):
                    with gr.Column():
                        gr.Markdown("""# Dataset3 Analysis Dashboard""")
                        
                        with gr.Row():
                            inputs = [gr.Dataframe(label="Dataset3", value=self.dataFrame3)]
                           
                            with gr.Column():
                                outputs = [gr.Textbox(label="Number of Rows"), gr.Textbox(label="Number of Columns"), gr.Dataframe(label="Attributes description")]
                        
                        btn = gr.Button("Submit")
                        btn.click(fn=self.infos_dataset, inputs=inputs, outputs=outputs)

                with gr.Tab("Discretization"):
                    with gr.Column():
                        gr.Markdown("""# Dataset3 Discretization""")
                        
                        with gr.Row():
                            method = [gr.Dropdown(["Equal-Width Discretization", "Equal-Frequency Discretization"], multiselect=False, label="Method", info="Select a method of Discretization :")]
                            attribute3 = gr.Dropdown([(f"{att}", i) for i, att in enumerate(self.dataFrame3.columns.tolist()) if i < 3], value=4, multiselect=False, label="Attributes", info="Select an attribute to discretize :")
                            bins_nbr = gr.Slider(2, 8, step=1, visible=False, label="Bins Number", info="Choose the number of bins :", value=8)
                           
                            def update_visibility_discretization(select_attribute):
                                    self.selected_attribute_dataset3 = select_attribute
                                    if select_attribute == 0:
                                        return {bins_nbr : gr.Slider(maximum=len(np.unique(self.dataset3[:, select_attribute].tolist())), visible=True)}
                                    if select_attribute == 1:
                                        return {bins_nbr : gr.Slider(maximum=len(np.unique(self.dataset3[:, select_attribute].tolist())), visible=True)}
                                    else:
                                        return {bins_nbr : gr.Slider(maximum=len(np.unique(self.dataset3[:, select_attribute].tolist())), visible=True)}
                                    
                            attribute3.change(update_visibility_discretization, inputs=[attribute3], outputs=bins_nbr)
                        with gr.Row():
                            output_dataset3 = [gr.Dataframe(label="Dataset3 after Discretization", headers=self.dataFrame3.columns.tolist())]
                            discretization_gallery = [gr.Gallery(label="Graphs", columns=(1,2))]

                        with gr.Row():
                            gr.ClearButton(method+[attribute3]+[bins_nbr])
                            btn_discr = gr.Button("Submit")
                            
                with gr.Tab("Frequent Itemsets and Association Rules"):
                    with gr.Column():
                        gr.Markdown("""# Dataset3 Frequent Itemsets and Association Rules""")
                        
                        with gr.Row():
                            transactions_table = gr.Dataframe(label="Transactions Table", headers=["Chosen Attribute", "Soil", "Crop", "Fertilizer"])
                            with gr.Column():    
                                inputs = [gr.Number(label="Minimal Support", value=0.01, step=0.01, minimum=0.01, maximum=1),
                                        gr.Number(label="Minimal Confidence", value=0.1, step=0.01, minimum=0.01, maximum=1),
                                        gr.Dropdown([(f"{m}", i) for i, m in enumerate(["Confidence", "Cosine", "Lift", "Jaccard", "Kulczynski"])], multiselect=False, label="Metric", info="Select a metric for association rules :")]
                            
                        with gr.Row():
                            freq_item = [gr.Dataframe(label="Frequent Itemsets")]
                            rules = [gr.Dataframe(label="Association Rules")]

                        with gr.Row():
                            gr.ClearButton(inputs)
                            btn = gr.Button("Submit")
                            btn.click(fn=self.FIL_general, inputs=[transactions_table]+inputs, outputs=rules+freq_item)
                            btn_discr.click(fn=self.discretization_plot, inputs=method+[attribute3]+[bins_nbr], outputs=output_dataset3+discretization_gallery+[transactions_table])

                with gr.Tab("Experimentations"):
                    with gr.Column():
                        gr.Markdown("""# Dataset3 Experimentations""")
                        
                        with gr.Row():  
                            inputs_exp = [gr.Number(label="Lower Bound of Minimal Support", minimum=0.0, maximum=0.3, value=0.01, step=0.01),
                                    gr.Number(label="Upper Bound of Minimal Support", minimum=0.0, maximum=0.3, value=0.2, step=0.01),
                                    gr.Number(label="Lower Bound of Minimal Confidence", minimum=0.0, maximum=0.3, value=0.01, step=0.01),
                                    gr.Number(label="Upper Bound of Minimal Confidence", minimum=0.0, maximum=0.3, value=0.2, step=0.01)]
                        with gr.Row():
                            experimentation_gallery = [gr.Gallery(label="Graphs", columns=(1,2))]

                        with gr.Row():
                            gr.ClearButton(inputs)
                            btn = gr.Button("Submit")
                            btn.click(fn=self.experimentation_plots, inputs=inputs_exp, outputs=experimentation_gallery)
                
                with gr.Tab("Recommender"):
                    with gr.Column():
                        gr.Markdown("""# Dataset3 Recommender""")
                        gr.Markdown(""" Insert new row :""")
                        with gr.Row():  
                            inputs = [gr.Textbox(label="Antecedent 1", value="Urea"),
                                gr.Textbox(label="Antecedent 2", value="29.283076923076926"),
                                gr.Textbox(label="Antecedent 3", value="Coconut"), 
                                gr.Radio(["Strict", "Not Strict"], label="Method")]
                        with gr.Row():
                            rec = [gr.Dataframe(label="Recommendation Table")]

                        with gr.Row():
                            gr.ClearButton(inputs)
                            btn = gr.Button("Submit")
                            btn.click(fn=self.recommendation, inputs=inputs, outputs=rec)


        self.demo_interface = demo

    def launch(self):
        self.demo_interface.launch()

app = App()
app.launch()