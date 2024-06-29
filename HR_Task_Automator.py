import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import io
# from langchain.llms import OpenAI
from transformers import GPT2Tokenizer
import re
import plotly.express as px
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader


# Sidebar for API Key and navigation
st.sidebar.title("Settings")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
selected_option = st.sidebar.radio("Select Task", [
    "Resume Job Hopper Identifier", 
    "Resume - Job Description Fit Identifier", 
    "Resume Data Miner"
])
st.sidebar.markdown("""
#### For Customized AI Applications Contact Me:
- LinkedIn: [John Embate](https://www.linkedin.com/in/john-william-embate-8759a4157)
- Email: [John Embate](mailto:johnembate0@gmail.com)
""", unsafe_allow_html=True)

st.title(selected_option)  # Use the selected option as the page title

# Initialize session state for data storage and file uploads
if 'data' not in st.session_state:
    st.session_state['data'] = []

if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []

# Function to prepare data for the spider chart
def prepare_data_for_chart(df, names):
    df_filtered = df[df['Name'].isin(names)]
    df_long = df_filtered.melt(id_vars='Name', value_vars=[col for col in df.columns if col != 'Name'],
                               var_name='variable', value_name='value')
    return df_long

# Function to generate a spider chart
def generate_spider_chart(data, title):
    # Generate the spider chart using Plotly Express
    fig = px.line_polar(data, r='value', theta='variable', line_close=True,
                        color='Name',  # Differentiating data by candidate
                        title=title, range_r=[0,100])
    fig.update_traces(fill='toself')
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True  # Ensure legend is shown
    )
    return fig 


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text() if page.extract_text() else ""
    return text

def count_tokens(text):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return len(tokenizer.encode(text))

def generate_response(uploaded_file, openai_api_key, job_hopper_threshold):
    query_text = f"""
    Please carefully analyze the resume text provided and extract specific information as listed below. Ensure that all job durations are presented in months (allows floating point numbers), and the response format is precisely adhered to.
    Please don't include internship on the overall analysis. Write NA if information is unavailable.
    1. Extract the full name of the applicant.
    2. Count and report the total number of jobs listed on the resume.
    3. Identify the job where the applicant spent the longest time, and provide the duration in months.
    4. Identify the name of the company (longest job duration).
    5. Determine the job where the applicant spent the shortest time, provide the duration in months.
    6. Identify the name of the company (shortest job duration).
    7. Calculate the average duration across all jobs, presented in months.
    
    -----------------------
    Provide the answers in a clear and structured format as follows (strictly):
    Name: <name>
    Total Number of Jobs: <number>
    Longest Time at a Job (Duration): <duration in months>
    Longest Time at a Job (Company): <company name>
    Shortest Time at a Job (Duration): <duration in months>
    Shortest Time at a Job (Company): <company name>
    Average Time at Jobs: <average duration in months>

    ----------------------
    """
    # Extract text from the PDF
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        prompt = query_text + text

        client = OpenAI(
                api_key = openai_api_key
        )
        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an HR manager which would like to get the following information from your applicants."}, 
            {"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0
    )
    return response.choices[0].message.content


def parse_response(response):
    result = {}
    lines = response.split('\n')
    for line in lines:
        if ': ' in line:
            key, value = line.split(': ', 1)
            result[key.strip()] = value.strip() if value.strip() else None
    return result

def get_first_numerical_value(text):
    # This function extracts the first numerical value from a given text string.
    if text is None:
        return None
    numbers = re.findall(r'\d+\.*\d*', text)  # This regex will find integers or decimals
    return float(numbers[0]) if numbers else None

# Conditional content based on selected feature
if selected_option == "Resume Job Hopper Identifier":


    job_hopper_threshold = st.number_input("Job Hopper Months Threshold Flag", min_value=1, value=5)
    uploaded_files = st.file_uploader("Upload Resume PDFs", accept_multiple_files=True, type="pdf", key="file_uploader")

    if "df_job_hopping_results" not in st.session_state:
        st.session_state["df_job_hopping_results"] = pd.DataFrame()

    if uploaded_files is not None:
        st.session_state.uploaded_files = uploaded_files
    
    if st.button('Run'):
        if not openai_api_key:
            st.warning("Please enter your OpenAI API Key to proceed.")
            st.stop()
        elif not st.session_state.uploaded_files:
            st.warning("Please upload at least one PDF file to analyze.")
            st.stop()
        else:
            all_data = []
            for pdf in st.session_state.uploaded_files:
                print(f"Running analysis for {pdf.name}")  # Debug output
                analysis_result = generate_response(pdf, openai_api_key, job_hopper_threshold)
                parsed_result = parse_response(analysis_result)
                # Debug print the parsed results
                print(f"Parsed Results for {pdf.name}: {parsed_result}")
                analysis_dict = {
                    "Name": parsed_result.get("Name", None),
                    "Total Number of Jobs": parsed_result.get("Total Number of Jobs", None),
                    "Longest Time at a Job (Duration)": parsed_result.get("Longest Time at a Job (Duration)", None),
                    "Longest Time at a Job (Company)": parsed_result.get("Longest Time at a Job (Company)", None),
                    "Shortest Time at a Job (Duration)": parsed_result.get("Shortest Time at a Job (Duration)", None),
                    "Shortest Time at a Job (Company)": parsed_result.get("Shortest Time at a Job (Company)", None),
                    "Average Time at Jobs": parsed_result.get("Average Time at Jobs", None),
                }
                # Compare the extracted duration with the job hopper threshold
                shortest_job_duration = analysis_dict.get("Shortest Time at a Job (Duration)")
                numerical_value = get_first_numerical_value(shortest_job_duration)
                if numerical_value is not None and numerical_value <= job_hopper_threshold:
                    analysis_dict["Is Job Hopper (1 or 0)"] = 1
                else:
                    analysis_dict["Is Job Hopper (1 or 0)"] = 0
                all_data.append(analysis_dict)
            
            df = pd.DataFrame(all_data)
            st.session_state['df_job_hopping_results'] = df
            
    
    if st.session_state['df_job_hopping_results'].empty == False:   
        st.write("Analysis Results")
        df = st.session_state['df_job_hopping_results']
        st.dataframe(df)
        # Download links for CSV
        csv = df.to_csv(index=False).encode('utf-8')
        csv_downloaded = st.download_button("Download as CSV", csv, 'resume_job_hoppers.csv', 'text/csv')
        excel = io.BytesIO()
        with pd.ExcelWriter(excel, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_downloaded = st.download_button("Download data as Excel", excel.getvalue(), 'resume_job_hoppers..xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        # Only show balloons if either download button is clicked
        if csv_downloaded or excel_downloaded:
            st.balloons()



elif selected_option == "Resume - Job Description Fit Identifier":
    
    # Input for job description
    job_description = st.text_area("Enter the Job Description", height=300)

    if "df_job_fit_results" not in st.session_state:
        st.session_state["df_job_fit_results"] = pd.DataFrame()

    # Dynamic criteria input with removal option
    if 'criteria' not in st.session_state:
        st.session_state['criteria'] = []
    
    with st.form("criteria_form"):
        # Custom function to display a help icon with a tooltip
        st.markdown("""
            <style>
                .tip-text {
                    font-size: 10px; /* Adjust the font size */
                    color: grey; /* Make the text grey to indicate it's a tip */
                    margin-bottom: 5px; /* Add some space below the text */
                }
            </style>
        """, unsafe_allow_html=True)

        crit_name = st.text_input("##### Criteria Name")
        st.markdown("##### Weight (1-100)")
        st.markdown("""Weight defines how much this criterion should contribute to the final decision, relative to other criteria. 
        The LLM will take these weights into account when evaluating applicants, ensuring that each criterion's importance is accurately reflected in the overall assessment.""")
        crit_weight = st.slider("Weight (1-100)", 1, 100)
        crit_description = st.text_area("##### Criteria Description", max_chars=200)
        st.markdown("Note: The applicant will be scored based on the range of (0-100)")
        add_criteria = st.form_submit_button("Add Criteria")
        

    if add_criteria:
        if len(crit_description.split()) <= 50:
            st.session_state['criteria'].append({
                'name': crit_name,
                'weight': crit_weight,
                'description': crit_description
            })
        else:
            st.error("Criteria description should not exceed 50 words.")

    # Display current criteria with option to remove
    if st.session_state['criteria']:
        for idx, crit in enumerate(st.session_state['criteria']):
            st.text(f"{crit['name']} - (Weight: {crit['weight']}): {crit['description']}")
            if st.button("Remove", key=f"remove_{idx}"):
                st.session_state['criteria'].pop(idx)
                st.experimental_rerun()

    uploaded_files = st.file_uploader("Upload Resume PDFs", accept_multiple_files=True, type="pdf", key="file_uploader_fit")

    if st.button('Analyze Resumes'):
        if not openai_api_key:
            st.warning("Please enter your OpenAI API Key to proceed.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF file to analyze.")
        elif not st.session_state.get('criteria'):
            st.warning("Please add at least one criterion.")
        else:
            all_data = []
            for uploaded_file in uploaded_files:
                text = extract_text_from_pdf(uploaded_file)
                prompt = f"""Please evaluate the following resume based on the job description and criteria provided. Score the applicant from the range of (0-100) based on the criteria given.
                The weight will be only be used as future basis for hiring the applicant, but you can use this as a guide for how critical a criteria should be scored by you.\n\n
                Resume text: {text}\n\nJob Description: {job_description}\n\n"""
                for idx, crit in enumerate(st.session_state['criteria'], start=1):
                    prompt += f"{idx}. {crit['name']} (Scoring: 0-100, Weight: {crit['weight']}): {crit['description']}\n"
                prompt += f"""\nSubmit the findings using the following format. Note: Please don't include an overall score. Just strictly follow the format and don't add anything.:\n
                ------------------------\n
                Name: <applicant's name>\n
                """
                for idx, crit in enumerate(st.session_state['criteria'], start=1):
                    prompt += f"{crit['name']}: <score number>\n"
                prompt+="-----------------------"

                client = OpenAI(
                api_key = openai_api_key
                )

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an HR manager tasked with critically scoring applicants based on different criteria and the job description."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1024,
                    temperature=0
                )

                
                scores = parse_response(response.choices[0].message.content)  # Implement parse_response to extract scores from the response
                print(f"Parsed Results for {uploaded_file.name}: {scores}")
                all_data.append(scores)

            df = pd.DataFrame(all_data)
            ### Dummy data sample
            # data = {
            #     "Name": ["Alejandra Talamante Garcia", "Aloha Reyes Baluyut", "Annie Gail Liangco Suba"],
            #     "Academic Achievement": [92, 88, 95],  # Hypothetical academic scores out of 100
            #     "Technical Skills": [68, 69, 87],  # Skills rating out of 10
            #     "Relevant Experience (years)": [72, 65, 93],  # Number of years of relevant experience
            #     "Leadership": [61, 70, 81]  # Binary indicator of leadership experience
            # }
            # df = pd.DataFrame(data)
            st.session_state['df_job_fit_results'] = df
            



    if st.session_state["df_job_fit_results"].empty == False:
        st.write("Scoring Results")
        df = st.session_state['df_job_fit_results']
        st.dataframe(df)
        # Download links for CSV
        csv = df.to_csv(index=False).encode('utf-8')
        csv_downloaded = st.download_button("Download data as CSV", csv, 'job_fit_analysis.csv', 'text/csv')
        excel = io.BytesIO()
        with pd.ExcelWriter(excel, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_downloaded = st.download_button("Download data as Excel", excel.getvalue(), 'job_fit_analysis.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        # Only show balloons if either download button is clicked
        if csv_downloaded or excel_downloaded:
            st.balloons()


    if st.session_state["df_job_fit_results"].empty == False:
        df = st.session_state['df_job_fit_results']
        # Button to trigger the chart generation
        st.markdown("### Candidates Evaluation and Comparison")
        selected_names = st.multiselect('Select Candidates', df['Name'])

        if st.button('Generate Chart'):
            if selected_names:
                # Prepare data for the selected candidates
                data_long = prepare_data_for_chart(df, selected_names)
                # Generate the spider chart
                fig = generate_spider_chart(data_long, 'Comparative Spider Chart for Selected Candidates')
                # Display the spider chart
                st.plotly_chart(fig, use_container_width=True)
                #st.balloons()
            else:
                st.warning("Please select at least one candidate to analyze.")

elif selected_option == "Resume Data Miner":

    if "df_resume_mine_results" not in st.session_state:
        st.session_state["df_resume_mine_results"] = pd.DataFrame()

    # Input for defining what data to mine
    if 'data_points' not in st.session_state:
        st.session_state['data_points'] = []

    with st.form("data_point_form"):
        new_data_point = st.text_input("Enter data point to extract from resume (e.g., 'Email', 'Name', 'Phone Number')")
        add_data_point = st.form_submit_button("Add Data Point")

    if add_data_point and new_data_point:
        st.session_state['data_points'].append(new_data_point)

    # Display current data points with option to remove
    if st.session_state['data_points']:
        for idx, point in enumerate(st.session_state['data_points']):
            st.text(f"Data Point {idx + 1}: {point}")
            if st.button(f"Remove", key=f"remove_{point}"):
                st.session_state['data_points'].remove(point)
                st.experimental_rerun()

    uploaded_files = st.file_uploader("Upload Resume PDFs", accept_multiple_files=True, type="pdf", key="file_uploader_miner")

    if st.button('Mine Resumes'):
        if not openai_api_key:
            st.warning("Please enter your OpenAI API Key to proceed.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF file to analyze.")
        elif not st.session_state.get('data_points'):
            st.warning("Please add at least one data point to extract.")
        else:
            all_results = []
            for uploaded_file in uploaded_files:
                text = extract_text_from_pdf(uploaded_file)
                prompt = f"""Extract the following information from the resume, enter NA if it doesn't exist:\n\nResume text: {text}\n\n"""
                for point in st.session_state['data_points']:
                    prompt += f"{point}\n"
                prompt += "\nPlease provide the information in a structured format.\n"
                for point in st.session_state['data_points']:
                    prompt += f"{point}: <information>\n"

                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Use "text-davinci-003" if you want the latest version
                    messages=[ 
                        {"role": "system", "content": "You are an HR manager tasked with mining important data information from the applicant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1024,
                    temperature=0
                )

                results = parse_response(response.choices[0].message.content)  # Access the text attribute directly
                all_results.append(results)

            result_df = pd.DataFrame(all_results)
            st.session_state['df_resume_mine_results'] = result_df
    
    if st.session_state["df_resume_mine_results"].empty == False:
        st.write("Extraction Results")
        df = st.session_state['df_resume_mine_results']
        st.dataframe(df)
        # Download links for CSV
        csv = df.to_csv(index=False).encode('utf-8')
        csv_downloaded = st.download_button("Download as CSV", csv, 'extracted_resume_information.csv', 'text/csv')
        excel = io.BytesIO()
        with pd.ExcelWriter(excel, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_downloaded = st.download_button("Download data as Excel", excel.getvalue(), 'extracted_resume_information.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        # Only show balloons if either download button is clicked
        if csv_downloaded or excel_downloaded:
            st.balloons()

