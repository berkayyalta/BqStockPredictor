#Stock Prediction Script, Made by Berkay H Y

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import statistics as stc
import yfinance as yf
import pandas as pd
import datetime

#Suprass the pandas warnings
import warnings
warnings.filterwarnings('ignore')

class GenerateDatasets():
    
    sections_x_vars = ['Open','Close','High','Low','Volume']    
    sections_period = [5,7,10,14,50,200]
    sections_degree = [2,3]
    rsi_ema = True

    #Methods for Array Generation
    def RowToArray(dataframe, row):
        #Generates an array including all columns in the row. row
        current_inspect = []
        for column_no in range(0, len(dataframe.columns), 1):
            current_inspect.append(dataframe.iloc[row, column_no])
        return current_inspect
            
    def GenerateXY(self, dataframe):
        #Dataframe must be including Y(Y-F1) at first column and X0(Y) at second column   
        xy_arrays = []
        dataframe = dataframe.dropna()            
        #Generate the X Array
        x_dataframe = dataframe.drop(dataframe.columns[0], axis=1)
        x_array = []
        for row_no_x in range(0, len(x_dataframe), 1):
            x_array.append(self.RowToArray(x_dataframe, row_no_x))        
        xy_arrays.append(x_array)           
        #Generate the Y array            
        y_array = []
        for row_no_y in range(0, len(dataframe), 1):
            if (dataframe.iloc[row_no_y, 0] > dataframe.iloc[row_no_y, 1]):
                y_array.append(1)
            else :
                y_array.append(0)
        xy_arrays.append(y_array)            
        #Return the Arrays
        return xy_arrays
    
    def GenerateP(dataframe):
        #Dataframe must be not processed because last row must be exist which is including NaN
        #Method doesn't support the forwards of X variables because NaN replacement
        dataframe = dataframe.drop(dataframe.columns[0], axis=1)
        last_row = dataframe.iloc[-1:]               
        p_array = []
        for column in range(0, len(last_row.columns), 1):
            p_array.append(last_row.iloc[0,column])  
        return p_array 
    
    #Methods for Dataframe Generation
    def CalculateRSI(self, dataframe, variable, rsi_period):       
        close_delta = dataframe[variable].diff()
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)            
        if self.rsi_ema == True:
            ma_up = up.ewm(com = rsi_period - 1, adjust=True, min_periods = rsi_period).mean()
            ma_down = down.ewm(com = rsi_period - 1, adjust=True, min_periods = rsi_period).mean()
        else:
            ma_up = up.rolling(window = rsi_period, adjust=False).mean()
            ma_down = down.rolling(window = rsi_period, adjust=False).mean()            
        rsi = ma_up / ma_down
        returnColumn = 100 - (100/(1 + rsi))
        return returnColumn
    
    def GenerateColumn(self, dataframe, variable):        
        #This method is generating VALID addjoint variable columns
        returnColumn = None              
        #Check for Exponential Variables
        for x_variable_ex in self.sections_x_vars:
            for degree in self.sections_degree:
                if (x_variable_ex + '-' + str(degree) == variable):
                    try :
                        returnColumn = dataframe[x_variable_ex]**degree  
                    except :
                        raise Exception(f'There is no {x_variable_ex} in your dataframe.')        
        #Check for Lagged Variables
        for x_variable_lag in self.sections_x_vars:
            for period_lag in self.sections_period:
                if (x_variable_lag + '-LAG' + str(period_lag) == variable):
                    try :
                        returnColumn = dataframe[x_variable_lag].shift(period_lag)  
                    except :
                        raise Exception(f'There is no {x_variable_lag} in your dataframe.')                      
        #Check for RSI Variables
        for x_variable_rsi in self.sections_x_vars:
            for period_rsi in self.sections_period:
                if (x_variable_rsi + '-RSI' + str(period_rsi) == variable):
                    #Try-Except is not necessary for this because it is already a method
                    returnColumn = self.CalculateRSI(self, dataframe, x_variable_rsi, period_rsi)            
        #Check for MA Variables
        for x_variable_ma in self.sections_x_vars:
            column_index_no = dataframe.columns.get_loc(x_variable_ma)
            for period_ma in self.sections_period:
                if (x_variable_ma + '-MA' + str(period_ma) == variable):
                    try :
                        returnColumn = dataframe.iloc[:,column_index_no].rolling(window=period_ma).mean()
                    except :
                        raise Exception(f'There is no {x_variable_ma} in your dataframe.')                   
        #More variable types can be added from here                                
        #Return the column
        if (returnColumn is None):
            raise Exception('Invalid input type for returnColumn, Variable : ' + variable)
        else :     
            return returnColumn
    
    def GenerateDataframe(self, stock, period, all_columns):
        #All_columns must be including all variables in order : Y(Y-F1), X0(Y), X1, X2, X3...
        #It returns a dataframe with rows including NaNs
        #Download the dataset
        today = datetime.datetime.now()
        d = datetime.timedelta(days = period)      
        start_date = str(today - d)[0:10]
        end_date = str(today)[0:10]         
        dataset = yf.download(stock, start=start_date, end=end_date)   
        #Create the dataframe and add the Y variable column
        y_variable = all_columns[0]
        dataframe = pd.DataFrame()
        dataframe[y_variable+'-F1'] = dataset[y_variable].shift(-1)       
        dataframe[y_variable] = dataset[y_variable]          
        #Add the X columns
        x_variables = []
        for x_variable_no in range(1, len(all_columns), 1):
            x_variables.append(all_columns[x_variable_no])  
            
        for x_variable in x_variables:
            if(x_variable in self.sections_x_vars):
                dataframe[x_variable] = dataset[x_variable]
            else :
                dataframe[x_variable] = self.GenerateColumn(self, dataset, x_variable)  
        return dataframe


class Calculation():
    
    k = 10   
    y_variable = 'Close'
    sklearn_scoring_type = 'accuracy'
    
    def Classifier(self, dataframe, model_validation_parts = None):
        if (model_validation_parts == None):
            #Generate Arrays
            xy_arrays = GenerateDatasets.GenerateXY(GenerateDatasets, dataframe)
            x_array, y_array = xy_arrays[0], xy_arrays[1]
            p_array = GenerateDatasets.GenerateP(dataframe)
            #Predict
            svm = SVC(kernel='rbf', gamma=10)
            #svm = SVC(kernel='rbf', random_state=1, gamma='scale', C=1.0)
            svm.fit(x_array, y_array)
            return svm.predict([p_array])
        else:            
            #Parts : x_train, y_train, x_test, y_test
            
            
            #Classifier
            x_train, y_train = model_validation_parts[0], model_validation_parts[1]
            svm = SVC(kernel='rbf', gamma=10)
            #Old : svm = SVC(kernel='rbf', random_state=1, gamma='scale', C=1.0)
            svm.fit(x_train, y_train)  
            
            #Score calculation   
            y_true = model_validation_parts[3] #Array type = Y Array
            y_pred = []            
            for inspect in model_validation_parts[2]: #Array type = X Array
                y_pred.append(svm.predict([inspect]))             
            if (self.sklearn_scoring_type == 'accuracy'):
                return accuracy_score(y_true, y_pred)
            elif (self.sklearn_scoring_type == 'balanced_accuracy'):
                return balanced_accuracy_score(y_true, y_pred)
            else :
                raise Exception('Wrong input type for scoring_type.')
    
    def ModelValidation(self, dataframe):

        scores = []        

        #It is necessary because all parts must be at same size
        dataframe = dataframe.dropna()

        #Drop excess rows and define part size
        for drop_excess in range(len(dataframe)%self.k):
            dataframe.drop(index=dataframe.index[0], axis=0, inplace=True)   
        part_size = int(len(dataframe)/self.k)  
        
        #Split the dataframe and append all parts to dataframe_parts
        dataframe_parts = []  
        for current_split in range(self.k):          
            current_dataframe = dataframe.iloc[:part_size]
            for remove_used_rows in range(part_size):
                dataframe.drop(index=dataframe.index[0], axis=0, inplace=True)              
            dataframe_parts.append(current_dataframe)
            
        #Sort dataframe parts
        sorted_dataframe_parts = []
        for dataframe_part_no in range(len(dataframe_parts)-1, -1, -1):
            sorted_dataframe_parts.append(dataframe_parts[dataframe_part_no])
            
        #Cross Validation
        cv_df_parts = sorted_dataframe_parts
        for current_validate_no in range(len(cv_df_parts)):
            
            #Generate Dataframes           
            train_df_parts = []
            for df_part_no in range(0, len(cv_df_parts), 1):
                if (df_part_no != current_validate_no):
                    train_df_parts.append(cv_df_parts[df_part_no])                    
            train_df = pd.concat(train_df_parts)            
            test_df = cv_df_parts[current_validate_no] 
            
            #Generate Arrays and Calculate
            xy_train = GenerateDatasets.GenerateXY(GenerateDatasets, train_df)
            xy_test = GenerateDatasets.GenerateXY(GenerateDatasets, test_df)
            cv_array = [xy_train[0], xy_train[1], xy_test[0], xy_test[1]]
            current_score = self.Classifier(self, None, cv_array)       
            
            scores.append(current_score)
            
        return stc.mean(scores)
    
    def Simulate(stock, lead, model, period):

        #Generate the Model    
        
        #TEMP
        model = ['Close']
        apx = ModelSelection.GeneratePossibleX(ModelSelection)
        for v in apx:
            model.append(v)
        
        dataframe = GenerateDatasets.GenerateDataframe(GenerateDatasets, stock, lead, model)
        x_model_array, y_model_array = GenerateDatasets.GenerateXY(GenerateDatasets, dataframe)[0], GenerateDatasets.GenerateXY(GenerateDatasets, dataframe)[1]
        svm = SVC(kernel='rbf', gamma=10)
        #Old : svm = SVC(kernel='rbf', random_state=1, gamma='scale', C=1.0)
        svm.fit(x_model_array, y_model_array)

        #X Check Dataframe
        x_check_dataframe = pd.DataFrame()
        x_check_dataframe['F1'], x_check_dataframe['Tod'] = dataframe.iloc[:, 0], dataframe.iloc[:, 1]
        x_check_dataframe = x_check_dataframe.dropna()
        x_check_dataframe = x_check_dataframe.tail(period)
        #X Inspect Dataframe
        x_inspect_dataframe = dataframe.dropna()
        x_inspect_dataframe = x_inspect_dataframe.tail(period)
        x_inspect_dataframe = x_inspect_dataframe.drop(dataframe.columns[0], axis=1)
        x_inspect_array = []
        for row_no in range(len(x_inspect_dataframe)):
            x_inspect_array.append(GenerateDatasets.RowToArray(x_inspect_dataframe, row_no))
               
        #Calculate
        if (period == len(x_inspect_array) and len(x_check_dataframe) == len(x_inspect_dataframe)):
            money = 100
            for day_no in range(len(x_inspect_array)):
                prediction = svm.predict([x_inspect_array[day_no]])
                
                if (prediction == 1 or prediction == [1]):
                    
                    buy = 100/(x_check_dataframe.iloc[day_no, 1])
                    sell = buy*(x_check_dataframe.iloc[day_no, 0])
                    money = sell
                    
                #To be edited
                print("Stock : ", stock, ", Day : ", day_no, " Amount : ", money)
                    
            return money
                    
        else :
            raise Exception('Wrong size for x_inspect_array or X Dataframes.',str(len(x_inspect_array)),str(len(x_inspect_dataframe)),str(len(x_check_dataframe)))
        
    
    def Predict(self, lead, calculation = 'all'):
        models = pd.read_csv('Models.csv')
        
        results = []
        if (calculation == 'all'):
            for stock_line in range(0, len(models), 1):  

                current_stock = models.iloc[stock_line, 0]       
                current_model = [self.y_variable]
                #Import the selected X variables
                for import_x_variable in range(1, len(models.columns), 1):
                    if (type(models.iloc[stock_line, import_x_variable]) == str):
                        current_model.append(models.iloc[stock_line, import_x_variable])

                    
                #Current calculation
                current_df = GenerateDatasets.GenerateDataframe(GenerateDatasets, current_stock, lead, current_model)
                current_result = self.Classifier(self, current_df)[0]
                current_score = self.ModelValidation(self, current_df)
                current_simulation = self.Simulate(current_stock, lead, current_model, 10)
                current_simulation5 = self.Simulate(current_stock, lead, current_model, 100)
                results.append([current_stock, current_result, current_score, current_simulation, current_simulation5])
                
                
        else :
            current_df = GenerateDatasets.GenerateDataframe(GenerateDatasets, 'ARCLK.IS', lead, ['Open','High','Low'])
            current_result = self.Classifier(current_df)
            current_score = self.ModelValidation(self, current_df)
            return current_result, current_score  
        
        return results
    
    
class ModelSelection():
    
    #General Fields
    y_variable = 'Close'
    sections_x_vars = ['Open','Close','High','Low','Volume']
    sections_degree = [2,3]
    sections_period = [5,7,10,14,50,200]
    max_x_num = 7
    min_x_num = 2
    
    def GeneratePossibleX(self):
        possible_x_variables = [] 
        for current_x_variable in self.sections_x_vars:      
            possible_x_variables.append(current_x_variable)   
            for degree in self.sections_degree:
                possible_x_variables.append(current_x_variable + f'-{degree}')             
            for period in self.sections_period:
                possible_x_variables.append(current_x_variable + f'-RSI{period}')
                possible_x_variables.append(current_x_variable + f'-MA{period}')
                possible_x_variables.append(current_x_variable + f'-LAG{period}') 
            #More X sections can be added from here
        try :
            possible_x_variables.remove(self.y_variable)
        except :
            raise Exception('Y Variable is not in the list.')
        return possible_x_variables
    
    def SelectVariable(stock, lead, model, x_variable_selections):
        results = []
        attempt = 1
        for x_variable in x_variable_selections:
            current_model = model
            current_model.append(x_variable)
            current_dataframe = GenerateDatasets.GenerateDataframe(GenerateDatasets, stock, lead, current_model)
            current_score = Calculation.ModelValidation(Calculation, current_dataframe)   
            results.append([x_variable, current_score])
            print(attempt)
            attempt += 1
        selected_result = ['None', 0.0]
        for result in results:
            if (result[1] > selected_result[1]):
                selected_result = result
        print('Added : ' + selected_result[0] + ' Score : ' + str(selected_result[1]))
        return selected_result[0]
    
    def SelectModel(self, stock, lead, x_variable_sections):
        selected_model = []
        #Calculate the current and previous model scores
        for add_new_variable in range(self.max_x_num):
            if (add_new_variable > self.min_x_num-1):   
                #Generate the current_vars array
                current_model_variables = [self.y_variable]
                for current_model_variable in selected_model:
                    current_model_variables.append(current_model_variable)            
                #Calculate the current model score
                current_model_dataframe = GenerateDatasets.GenerateDataframe(GenerateDatasets, stock, lead, current_model_variables)
                current_model_score = Calculation.ModelValidation(Calculation, current_model_dataframe)      
                #Calculate the previous model score
                previous_model_vars = []               
                for variable_no in range(len(current_model_variables)-1):
                    previous_model_vars.append(current_model_variables[variable_no])                   
                previous_model_dataframe = GenerateDatasets.GenerateDataframe(GenerateDatasets, stock, lead, previous_model_vars)
                previous_model_score = Calculation.ModelValidation(Calculation, previous_model_dataframe)           
            else :
                current_model_score = 1
                previous_model_score = 0
        

            
            #Decide to continue
            if (previous_model_score < current_model_score):
                
                current_model = [self.y_variable]
                for model_var in selected_model:
                    current_model.append(model_var)
                best_section = self.SelectVariable(stock, lead, current_model, x_variable_sections)
                selected_model.append(best_section)
                x_variable_sections.remove(best_section)
                
            else :
                new_selected_model = []
                for selected_variable in range(len(selected_model)-1):
                    new_selected_model.append(selected_model[selected_variable])
                selected_model = new_selected_model
                break
        
        #Return the selected model
        return selected_model
            
            

        
    def SaveModel(self, stock, lead):
        possible_x_variables = self.GeneratePossibleX(self)
        best_model = self.SelectModel(self, stock, lead, possible_x_variables)           
        #Generate the model text to be added to the file (Model = [STOCK, X1, X2, X3...])
        print('Model for Stock : ', stock, 'is complete, the model is : ', best_model)   
        model_text = f'{stock},'
        for variable_no in range(0, len(best_model), 1):
            if (variable_no == len(best_model)-1):
                model_text += best_model[variable_no]+'\n'
            else :
                model_text += best_model[variable_no]+','                     
        #Decide to Replace or Add the new row to Models.csv
        try:
            with open('Models.csv', 'r') as models_csv_read:
                rows = models_csv_read.readlines()     
                with open('Models.csv', 'w') as models_csv_write:
                    for row in rows:
                        if row.find(stock) == -1:
                            models_csv_write.write(row)
            models = open('Models.csv','a')
            models.write(model_text)
            models.close()          
        except :
            models = open('Models.csv','a')
            models.write(model_text)
            models.close()
                



    