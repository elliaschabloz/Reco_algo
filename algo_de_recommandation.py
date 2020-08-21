#Importing libraries

import numpy as np
import pandas as pd
import operator
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.sparse.linalg import svds

#read csv files

df = pd.read_csv('./ratings.csv', sep=',')
df = shuffle(df)
train, test = train_test_split(df, test_size=0.3)

#creation des deux matrices
matrice_70= train.pivot_table(index='userId', columns='movieId',values='rating')
matrice_30= test.pivot_table(index='userId', columns='movieId',values='rating')


valeurs_matrice_70 = matrice_70.values
valeurs_matrice_30 = matrice_30.values

valeurs_matrice_70_t = valeurs_matrice_70.transpose()

colonnes_matrice_70 = matrice_70.columns.tolist()
colonnes_matrice_30 = matrice_30.columns.tolist()

lignes_matrice_70 = matrice_70.index.tolist()
lignes_matrice_30 = matrice_30.index.tolist()

moyenne = np.nanmean(valeurs_matrice_70)
dimensions = valeurs_matrice_70.shape



def global_baseline_predictor(valeurs_matrice_70,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30):
	erreur_rmse = 0
	erreur_mae = 0
	cardinal_erreur = 0

	#moyenne = np.nanmean(valeurs_matrice_70)
	for row in range(dimensions[0]):
		for column in range(dimensions[1]):
			if np.isnan(valeurs_matrice_70[row][column]):
				try:
					l = lignes_matrice_30.index(lignes_matrice_70[row])
					c = lignes_matrice_30.index(colonnes_matrice_70[column])
					if not np.isnan(valeurs_matrice_30[l][c]):
						erreur_mae += abs(valeurs_matrice_30[l][c] - moyenne)
						erreur_rmse += (valeurs_matrice_30[l][c] - moyenne) ** 2
						cardinal_erreur += 1
				except ValueError:
					continue

	return (((erreur_mae) / cardinal_erreur) , sqrt(erreur_rmse) / cardinal_erreur)

def user_baseline_predictor(valeurs_matrice_70,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30):
	erreur_rmse = 0
	erreur_mae = 0
	cardinal_erreur = 0

	for row in range(dimensions[0]):
		mean_row = np.nanmean(valeurs_matrice_70[row])
		for column in range(dimensions[1]):
			if np.isnan(valeurs_matrice_70[row][column]):
				try:
					l = lignes_matrice_30.index(lignes_matrice_70[row])
					c = lignes_matrice_30.index(colonnes_matrice_70[column])
					if not np.isnan(valeurs_matrice_30[l][c]):
						erreur_mae += abs(valeurs_matrice_30[l][c] - mean_row)
						erreur_rmse += (valeurs_matrice_30[l][c] - mean_row) ** 2
						cardinal_erreur += 1
				except ValueError:
					continue
	return (((erreur_mae) / cardinal_erreur) , sqrt(erreur_rmse) / cardinal_erreur)

def item_baseline_predictor(valeurs_matrice_70,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30):
	erreur_rmse = 0
	erreur_mae = 0
	cardinal_erreur = 0
	mean_column_list = np.nanmean(valeurs_matrice_70 , axis=0)
	for column in range(dimensions[1]):
		mean_column = mean_column_list[column]
		for row in range(dimensions[0]):
			if np.isnan(valeurs_matrice_70_t[row][column]):
				try:
					l = lignes_matrice_30.index(lignes_matrice_70[column])
					c = lignes_matrice_30.index(colonnes_matrice_70[row])
					if not np.isnan(valeurs_matrice_30[l][c]):
						erreur_mae += abs(valeurs_matrice_30[l][c] - mean_column)
						erreur_rmse += (valeurs_matrice_30[l][c] - mean_column) ** 2
						cardinal_erreur += 1
				except ValueError:
					continue
	return (((erreur_mae) / cardinal_erreur) , sqrt(erreur_rmse) / cardinal_erreur)

# def user_bias(row , moyenne):
# 	somme = 0
# 	cardinal = 0
# 	for r in row:
# 		if not np.isnan(r):
# 			somme += (r - moyenne)
# 			cardinal += 1

# 	return (somme / cardinal)

# def item_bias(valeurs_matrice_70, column, moyenne):
# 	somme = 0
# 	cardinal = 0
# 	for row in range(len(valeurs_matrice_70)):
# 		if not np.isnan(valeurs_matrice_70[row][column]):
# 			somme += (valeurs_matrice_70[row][column] - moyenne - user_bias(valeurs_matrice_70[row], moyenne))
# 			cardinal += 1
# 	return somme / cardinal

def user_item_bias_model(valeurs_matrice_70,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30):
	erreur_rmse = 0
	erreur_mae = 0
	cardinal_erreur = 0

	#moyenne = np.nanmean(valeurs_matrice_70)

	for row in range(dimensions[0]):
		for column in range(dimensions[1]):
			if np.isnan(valeurs_matrice_70[row][column]):
				try:
					l = lignes_matrice_30.index(lignes_matrice_70[row])
					c = lignes_matrice_30.index(colonnes_matrice_70[column])
					if not np.isnan(valeurs_matrice_30[l][c]):
						user_b = np.nanmean([x  - moyenne for x in valeurs_matrice_70[row]])
						item_b = np.nanmean([x  - (moyenne - user_b) for x in valeurs_matrice_70_t[column]])
						bias = moyenne + user_b + item_b
						erreur_mae += abs(valeurs_matrice_30[l][c] - bias)
						erreur_rmse += (valeurs_matrice_30[l][c] - bias) ** 2
						cardinal_erreur += 1
				except ValueError:
					continue
	return (((erreur_mae) / cardinal_erreur) , sqrt(erreur_rmse) / cardinal_erreur)


def sim_user(row1, row2):
	#vector cosine plus efficace
	avrg_usr1 = np.nanmean(row1)
	avrg_usr2 = np.nanmean(row2)

	numerator = 0
	denominator1 = 0
	denominator2 = 0

	for index in range(len(row1)):
		if not np.isnan(row1[index]):
			denominator1 += (row1[index] - avrg_usr1) ** 2
			if not np.isnan(row2[index]):
				numerator += (row1[index] - avrg_usr1) * (row2[index] - avrg_usr2)
				denominator2 += (row2[index] - avrg_usr2) ** 2		
			else:
				denominator2 += (avrg_usr2) ** 2
		else:
			denominator1 += (avrg_usr1) ** 2
			if not np.isnan(row2[index]):
				denominator2 += (row2[index] - avrg_usr2) ** 2
			else:
				denominator2 += (avrg_usr2) ** 2

	return numerator / (sqrt(denominator1) * sqrt(denominator2))

def sim_item(item1, item2):
	#avec transposee pour eviter calculs longs

	avrg_item1 = np.nanmean(np.nanmean(item1))
	avrg_item2 = np.nanmean(np.nanmean(item2))

	numerator = 0
	denominator1 = 0
	denominator2 = 0

	for index in range(len(item1)):
		if not np.isnan(item1[index]):
			denominator1 += (item1[index] - avrg_item1) ** 2
			if not np.isnan(item2[index]):
				numerator += (item1[index] - avrg_item1) * (item2[index] - avrg_item2)
				denominator2 += (item2[index] - avrg_item2) ** 2		
			else:
				denominator2 += (avrg_item2) ** 2
		else:
			denominator1 += (avrg_item1) ** 2
			if not np.isnan(item2[index]):
				denominator2 += (item2[index] - avrg_item2) ** 2
			else:
				denominator2 += (avrg_item2) ** 2

	return numerator / (sqrt(denominator1) * sqrt(denominator2))

def neighborhood_user(k,user,valeurs_matrice_70) :
	#user is an index
	N = [(0,-100)] * k
	for index in range(len(valeurs_matrice_70)):
		if index != user:
			minimum = min(N,key=operator.itemgetter(1))
			sim = sim_user(valeurs_matrice_70[user],valeurs_matrice_70[index])
			if sim > minimum[1]:
				N.remove(minimum)
				N.append((index,sim))
	return N

def neighborhood_item(k,item,valeurs_matrice_70):
	#item est un indice
	N = [(0,-100)] * k
	for index in range(len(valeurs_matrice_70)):
		if index != item:
			minimum = min(N,key=operator.itemgetter(1))
			sim = sim_item(valeurs_matrice_70[item],valeurs_matrice_70[index])
			if sim > minimum[1]:
				N.remove(minimum)
				N.append((index,sim))
	return N

def knn_UU(valeurs_matrice_70,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30):
	K = 30
	erreur_rmse = 0
	erreur_mae = 0
	cardinal_erreur = 0

	for row in range(dimensions[1]):
		neighhor = neighborhood_user(K,row,valeurs_matrice_70)
		for column in range(dimensions[1]):
			if np.isnan(valeurs_matrice_70[row][column]):
				try:
					l = lignes_matrice_30.index(lignes_matrice_70[row])
					c = lignes_matrice_30.index(colonnes_matrice_70[column])
					if not np.isnan(valeurs_matrice_30[l][c]):
						neighhor_rated = []
						for element in neighhor:
							if not np.isnan(valeurs_matrice_70[element[0]][column]):
								neighhor_rated.append(element)
						if len(neighhor_rated) != 0:
							avrg_usr = np.nanmean(valeurs_matrice_70[row])
							somme_num = 0
							somme_dem = 0
							for element in neighhor_rated:
								somme_num += element[1] * (valeurs_matrice_70[element[0]][column] - avrg_usr)
								somme_dem += abs(element[1])
							prediction = avrg_usr + (somme_num / somme_dem)
							erreur_mae += abs(valeurs_matrice_30[l][c] - prediction)
							erreur_rmse += (valeurs_matrice_30[l][c] - prediction) ** 2
							cardinal_erreur += 1
				except ValueError:
					continue
	return (((erreur_mae) / cardinal_erreur) , sqrt(erreur_rmse) / cardinal_erreur)

def knn_II(valeurs_matrice_70_t,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30):
	K = 30
	erreur_rmse = 0
	erreur_mae = 0
	cardinal_erreur = 0

	for row in range(dimensions[1]):
		neighhor = neighborhood_item(K,row,valeurs_matrice_70_t)
		for column in range(dimensions[0]):
			if np.isnan(valeurs_matrice_70_t[row][column]):
				try:
					l = lignes_matrice_30.index(lignes_matrice_70[column])
					c = lignes_matrice_30.index(colonnes_matrice_70[row])
					if not np.isnan(valeurs_matrice_30[l][c]):						
						neighhor_rated = []
						for element in neighhor:
							if not np.isnan(valeurs_matrice_70_t[element[0]][column]):
								neighhor_rated.append(element)
						if len(neighhor_rated) != 0:
							avrg_item = np.nanmean(valeurs_matrice_70_t[row])
							somme_num = 0
							somme_dem = 0
							for element in neighhor_rated:
								somme_num += element[1] * (valeurs_matrice_70_t[element[0]][column] - avrg_item)
								somme_dem += abs(element[1])
							prediction = avrg_item + (somme_num / somme_dem)

							erreur_mae += abs(valeurs_matrice_30[l][c] - prediction)
							erreur_rmse += (valeurs_matrice_30[l][c] - prediction) ** 2
							cardinal_erreur += 1
				except ValueError:
					continue
	return (((erreur_mae) / cardinal_erreur) , sqrt(erreur_rmse) / cardinal_erreur)


def SVD(valeurs_matrice_70,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30):
	#moyenne = np.nanmean(valeurs_matrice_70)
	matrice_remplie = np.copy(valeurs_matrice_70)
	matrice_remplie[np.isnan(matrice_remplie)] = np.nanmean(valeurs_matrice_70)
	matrice_remplie = [[y - moyenne for y in x] for x in matrice_remplie]
	U, S, Vt = np.linalg.svd(matrice_remplie, full_matrices=True)
	S = np.diag(S)
	X = np.matmul(U,np.sqrt(S))
	Y = np.matmul(np.transpose(Vt),np.sqrt(S))


	erreur_rmse = 0
	erreur_mae = 0
	cardinal_erreur = 0
	
	for row in range(dimensions[0]):
		for column in range(dimensions[1]):
			if np.isnan(valeurs_matrice_70[row][column]):
				try:
					l = lignes_matrice_30.index(lignes_matrice_70[row])
					c = lignes_matrice_30.index(colonnes_matrice_70[column])
					if not np.isnan(valeurs_matrice_30[l][c]):
						prediction = moyenne + np.matmul(X[row],Y[column])
						erreur_mae += abs(valeurs_matrice_30[l][c] - prediction)
						erreur_rmse += (valeurs_matrice_30[l][c] - prediction) ** 2
						cardinal_erreur += 1
				except ValueError:
					continue
	return (((erreur_mae) / cardinal_erreur) , sqrt(erreur_rmse) / cardinal_erreur)

def approximation_bas_rang(valeurs_matrice_70,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30):
	#moyenne = np.nanmean(valeurs_matrice_70)
	matrice_remplie = np.copy(valeurs_matrice_70)

	matrice_remplie[np.isnan(matrice_remplie)] = np.nanmean(valeurs_matrice_70)
	matrice_remplie = [[y - moyenne for y in x] for x in matrice_remplie]

	U, S, Vt = svds(matrice_remplie, k=40)

	S = np.diag(S)
	X = np.matmul(U,np.sqrt(S))
	Y = np.matmul(np.transpose(Vt),np.sqrt(S))

	erreur_rmse = 0
	erreur_mae = 0
	cardinal_erreur = 0

	for row in range(dimensions[0]):
		for column in range(dimensions[1]):
			if np.isnan(valeurs_matrice_70[row][column]):
				try:
					l = lignes_matrice_30.index(lignes_matrice_70[row])
					c = lignes_matrice_30.index(colonnes_matrice_70[column])
					if not np.isnan(valeurs_matrice_30[l][c]):
						prediction = moyenne + np.matmul(X[row],Y[column])
						#prediction = np.matmul(X[row],Y[column])

						erreur_mae += abs(valeurs_matrice_30[l][c] - prediction)
						erreur_rmse += (valeurs_matrice_30[l][c] - prediction) ** 2
						cardinal_erreur += 1
				except ValueError:
					continue
	return (((erreur_mae) / cardinal_erreur) , sqrt(erreur_rmse) / cardinal_erreur)

# GBP = global_baseline_predictor(valeurs_matrice_70,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30)
# print("Mean Absolute Error of global baseline predictor: {0} \nRoot Mean Square Error of global baseline predictor {1}\n".format(GBP[0],GBP[1]))

# UBP = user_baseline_predictor(valeurs_matrice_70,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30)
# print("Mean Absolute Error of user baseline predictor: {0} \nRoot Mean Square Error of user baseline predictor {1}\n".format(UBP[0],UBP[1]))

IBP = item_baseline_predictor(valeurs_matrice_70,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30)
print("Mean Absolute Error of item baseline predictor: {0} \nRoot Mean Square Error of item baseline predictor {1}\n".format(IBP[0],IBP[1]))

# UIBM = user_item_bias_model(valeurs_matrice_70,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30)
# print("Mean Absolute Error of user item bias model : {0} \nRoot Mean Square Error of user item bias model  {1}\n".format(UIBM[0],UIBM[1]))

# KNNU = knn_UU(valeurs_matrice_70,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30)
# print("Mean Absolute Error of KNN user : {0} \nRoot Mean Square Error of KNN user {1}\n".format(KNNU[0],KNNU[1]))

#KNNI = knn_II(valeurs_matrice_70_t,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30)
#print("Mean Absolute Error of KNN item : {0} \nRoot Mean Square Error of KNN item {1}\n".format(KNNI[0],KNNI[1]))

# SvD = SVD(valeurs_matrice_70,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30)
# print("Mean Absolute Error of SVD : {0} \nRoot Mean Square Error of SVD {1}\n".format(SvD[0],SvD[1]))

# ABR = approximation_bas_rang(valeurs_matrice_70,valeurs_matrice_30,colonnes_matrice_70,colonnes_matrice_30,lignes_matrice_70,lignes_matrice_30)
# print("Mean Absolute Error of ABR : {0} \nRoot Mean Square Error of ABR {1}\n".format(ABR[0],ABR[1]))
