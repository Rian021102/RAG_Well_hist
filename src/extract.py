import os
import re
import pandas as pd
from PyPDF2 import PdfReader

def extract_data_from_pdfs(folder_path):
    # Initialize list to hold dataframes for each file
    df_list = []
    
    # List all PDF files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            # Extract wellbore name from the filename
            wellbore_name = filename.split('_')[0] + '_' + filename.split('_')[1]+filename.split('_')[2]+filename.split('_')[3]+filename.split('_')[4]
            
            # Load the PDF file
            reader = PdfReader(os.path.join(folder_path, filename))
            first_page = reader.pages[0]
            text = first_page.extract_text()
            
            # Regular expression to find the Report creation time
            report_time_pattern = re.compile(r"Report creation time:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2})")
            report_creation_time = report_time_pattern.search(text).group(1) if report_time_pattern.search(text) else "Unknown Time"
            
            # Initialize lists to hold extracted data
            start_times, end_times, end_depths, activities, states, remarks = [], [], [], [], [], []
            
            # Regular expression to match the operations data pattern
            operations_pattern = re.compile(r"(\d{2}:\d{2})\s+(\d{2}:\d{2})\s+(\d+)\s+(.*?)\s+(ok|fail)\s+(.*?)\n")
            
            # Extract data using the defined pattern
            for match in operations_pattern.finditer(text):
                start_times.append(match.group(1))
                end_times.append(match.group(2))
                end_depths.append(int(match.group(3)))
                activities.append(match.group(4))
                states.append(match.group(5))
                remarks.append(match.group(6))
            
            # Create a DataFrame for this file
            data = {
                "Wellbore": [wellbore_name] * len(start_times),
                "Report Creation Time": [report_creation_time] * len(start_times),
                "Start Time": start_times,
                "End Time": end_times,
                "End Depth mMD": end_depths,
                "Main - Sub Activity": activities,
                "State": states,
                "Remark": remarks
            }
            df = pd.DataFrame(data)
            df_list.append(df)
    
    # Concatenate all DataFrames into one, sorting by 'Wellbore'
    if df_list:
        final_df = pd.concat(df_list)
        final_df.sort_values(by='Wellbore', inplace=True)
        return final_df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no files were processed

# Example usage:
df = extract_data_from_pdfs("/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/data")
#save the data to csv
df.to_csv('/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/data/drilling_report_comp.csv', index=False)
print(df)
