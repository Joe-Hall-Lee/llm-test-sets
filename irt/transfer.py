import chardet
import csv
import json
import os
import jsonlines


# mmlu需要删除的数据，索引为类别，值为索引列表
loss_index = {'anatomy': [119, 120, 121, 122, 123, 124, 125], 
              'college_biology': [34], 
              'college_chemistry': [79], 
              'college_medicine': [25], 
              'computer_security': [77], 
              'high_school_biology': [100], 
              'high_school_us_history': [57], 
              'philosophy': [173, 208, 212], 
              'prehistory': [214], 
              'professional_accounting': [238], 
              'professional_psychology': [47, 94, 291, 292, 412, 449, 453, 585]} 


# 获得文件编码
def get_encoding(file):
    # 二进制方式读取，获取字节数据，检测类型
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']


# csv文件处理为pyrit数据
# start是第一个模型处理列的索引
def csv_pyrit(in_path,out_path,start):
    data = {}
    model_list=[]
    end = 0
    encode = get_encoding(in_path)
    with open(in_path,encoding=encode) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for (i,row) in enumerate(csv_reader):   # 将csv 文件中的数据保存到data中
            if i == 0:
                for j in range(len(row)):
                    if row[j] == '':
                        break
                end = j
                if row[end] != '':
                    end = end + 1
                row = row[start:end]
                for item in row:
                    model_list.append(item)
                    data[item] = []
            else:
                row = row[start:end]
                for j in range(len(row)):
                    data[model_list[j]].append(row[j])
        
    data_form = []
    for i in range(len(model_list)):
        list1 = {}
        list1['subject_id'] = model_list[i]
        results = data[model_list[i]]
        response = {}
        for (i,item) in enumerate(results):
            index_str = 'q' + str(i+1)
            response[index_str] = int(item)
        list1['responses'] = response
        data_form.append(list1)

    with jsonlines.open(out_path, 'w') as w:
        for item in data_form:
            w.write(item)



# json文件转化为jsonlines
def convert():
    path = "pyrit_data" #文件夹目录
    files= os.listdir(path) #得到文件夹下的所有文件名称
    for file in files: #遍历文件夹
        file = path + '/' + file
        new_file = file[:-4]+'jsonlines'
        with open(file,'r',encoding='utf-8') as load_f:
            data = json.load(load_f)
        with jsonlines.open(new_file, 'w') as w:
            for item in data:
                w.write(item)
        





# json文件夹转化为pyrit数据
# path 输入文件夹名
# subject_id 模型名称
def json_files_pyrit(path,out_path,subject_id):
    res_list = []
    files= os.listdir(path) #得到文件夹下的所有文件名称
    for file in files: #遍历文件夹
        file = path +'/' + file
        with open(file,'r',encoding='utf-8') as load_f:
            data = json.load(load_f)
        data = data['details']
        length = len(data)
    
        for i in range(length):
            item = data[str(i)]
            res = item['is_correct']  # 不同数据属性不同，is_correct/correct
            if res:
                res_list.append(1)
            else:
                res_list.append(0)
    print(len(res_list))
    list1 = {}
    list1['subject_id'] = subject_id
    response = {}
    for (i,item) in enumerate(res_list):
        index_str = 'q' + str(i+1)
        response[index_str] = int(item)
    list1['responses'] = response

    with jsonlines.open(out_path, 'a') as w:
        w.write(list1)

   

# json文件转化为pyrit数据
# subject_id 模型名称
def json_pyrit(in_path,out_path,subject_id):
    with open(in_path,'r',encoding='utf-8') as load_f:
        data = json.load(load_f)
    data = data['details']
    length = len(data)
    res_list = []
    for i in range(length):
        item = data[str(i)]
        res = item['correct']  # 不同数据属性不同，is_correct/correct
        if res:
            res_list.append(1)
        else:
            res_list.append(0)
    
    list1 = {}
    list1['subject_id'] = subject_id
    response = {}
    for (i,item) in enumerate(res_list):
        index_str = 'q' + str(i+1)
        response[index_str] = int(item)
    list1['responses'] = response

    with jsonlines.open(out_path, 'a') as w:
        w.write(list1)






# in_path='result/gpqa_diamond_results.csv'
# out_path = 'pyrit_data/gpqa_diamond.jsonlines'
# start = 5
# csv_pyrit(in_path,out_path,start)
# convert()
# in_path='result/deepseek_v3'
# out_path = 'pyrit_data/mmmlu.jsonlines'
# subject_id = 'deepseek_v3'
# json_files_pyrit(in_path,out_path,subject_id)

model_list = ['deepseek-v3','hunyuan-turbo','spark4.0-ultra']
for model in model_list:
    in_path = 'result/GPQA_diamond_' + model + '.json'
    out_path = 'pyrit_data/gpqa_diamond.jsonlines'
    json_pyrit(in_path,out_path,model)


