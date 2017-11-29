from Run_linear_model import run_linear_model
from data_extarcting.DataFilesObj import DataFilesObj
import itertools


class LeaveNOutObj:
    N = 0
    data_file_obj = None
    acc_result_list = []
    csv_files_dir  = ""

    def __init__(self, N, data_file_obj, csv_files_dir  = ""):
        self.N = N
        self.data_file_obj = data_file_obj
        self.csv_files_dir= csv_files_dir

    def run_leave_N_out (self):
        combinations_iter = self.get_combination ()
        for combination in combinations_iter:
            training_data, test_data = self.split_data (combination)
            self.data_file_obj.create_csv_files_with_premade_data(training_data,test_data,self.csv_files_dir)
            results = run_linear_model(self.csv_files_dir)
            self.acc_result_list.append(results[0])
        return (sum(self.acc_result_list) / float(len(self.acc_result_list)))


    def get_combination (self):
        length_list = range(len(self.data_file_obj.data_list))
        iter = itertools.combinations (length_list, self.N)
        return iter

    def split_data(self, combination):
        test_list = []
        training_list = []
        for index in range(len(self.data_file_obj.data_list)):
            if index in combination:
                test_list.append(self.data_file_obj.data_list[index])
            else:
                training_list.append(self.data_file_obj.data_list[index])


d= DataFilesObj()
d.choose_features_by_column_num([18,19],0)
test = LeaveNOutObj (7,d,"S&T")
test.run_leave_N_out()
