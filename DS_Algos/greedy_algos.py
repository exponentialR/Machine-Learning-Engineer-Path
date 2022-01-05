def get_maximum_jobs(data):
    """
    You were given a list of jobs with start time and end times. You can only do one job at any time. 
    To complete a job, you have to start it at the start time and work until completion. 
    What is the maximum amount of jobs that can be completed? (Activity Selection)
    """

    #if the length of data is 0 return 0
    if len(data) == 0:
        return 0

    #sort the job by finishing time 
    jobs_sorted = sorted(data, key = lambda x: x[1])
    
    #start with the firtst job 
    current_job = jobs_sorted[0]
    jobs_completed = 1

    #iterate for all the jobs and keep adding it in the count if the new job start after the end of previous
    for i in range(1, len(jobs_sorted)):
        nex_job = jobs_sorted[i]
        if current_job[1]<nex_job[0]:
            current_job = nex_job
            jobs_completed +=1
    return jobs_completed


def fill_knapsack(data, capacity):

    if len(data) == 0 or capacity <=0:
        return 0
    #list to hold value weight ratio

    value_weight_ratio = list()

    #Get the value weight ratio

    for item in data:
        value_weight_ratio.append((item[0]/item[0], item[1]))
    #list is sorted based on value weight ratio
    values_sorted = sorted(value_weight_ratio, key = lambda x: x[0], reverse = True)

    #Initial item count is 0 and value is 0
    current_item = 0
    total_value = 0
    while capacity > 0 and current_item < len(values_sorted):
        if (capacity-values_sorted[current_item][2]) >=0:
            capacity = capacity-values_sorted[current_item][2]
            total_value += values_sorted[current_item][1]

        else:
            total_value += capacity * values_sorted[current_item][0]
            capacity-=values_sorted[current_item][2]
            break

        current_item+=1