from Run_linear_model import run_linear_model

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
        permutation_list = self.get_permutations ()
        for permutation in permutation_list:
            training_data, test_data = self.split_data (permutation)
            self.data_file_obj.create_csv_files_with_premade_data(training_data,test_data,self.csv_files_dir)
            results = run_linear_model(self.csv_files_dir)
            self.acc_result_list.append(results[0])
        return (sum(self.acc_result_list) / float(len(self.acc_result_list)))


    def get_permutations (self):
        pass

    def split_data (self, permutation):
        pass