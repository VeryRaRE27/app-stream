import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import cvxpy as cp
import warnings
import plotly.express as px
import plotly.graph_objects as go

# Define constants
employee_names = {1: 'Yusheen', 2: 'Roxanne', 3: 'Camilla', 4: 'Sue', 5: 'Meli', 6: 'Mpho', 7: 'John', 8: 'Alice'}
num_shifts = 4

class ScheduleOptimizer:
    def __init__(self, employee_ids, num_shifts):
        self.employee_ids = employee_ids
        self.num_employees = len(employee_ids)
        self.num_shifts = num_shifts
        self.availability_model = None

    def train_availability_model(self, X_train, y_train):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        self.availability_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000)
        self.availability_model.fit(X_train_scaled, y_train)

    def generate_feature_vector(self, date):
        return np.random.rand(self.num_employees, 10)

    def predict_availability(self, date):
        X_test = self.generate_feature_vector(date)

        if self.availability_model is None:
            raise Exception("The availability model is not trained yet.")

        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)

        probabilities = self.availability_model.predict_proba(X_test_scaled)

        if probabilities.shape[1] != self.num_shifts:
            probabilities = np.hstack([probabilities, np.zeros((self.num_employees, self.num_shifts - probabilities.shape[1]))])

        return probabilities

    def solve_optimization_problem(self, availability_predictions):
        shift_vars = cp.Variable((self.num_employees, self.num_shifts), boolean=True)

        constraints = [cp.sum(shift_vars[:, j]) == 1 for j in range(self.num_shifts)]
        constraints += [cp.sum(shift_vars[i, :]) <= 1 for i in range(self.num_employees)]

        for i in range(self.num_employees):
            for j in range(self.num_shifts - 1):
                constraints.append(shift_vars[i, j] + shift_vars[i, j + 1] <= 1)
            constraints.append(cp.sum(shift_vars[i, :]) * 8 <= 45 * 7)

        objective = cp.Maximize(cp.sum(cp.multiply(availability_predictions, shift_vars)))

        problem = cp.Problem(objective, constraints)
        problem.solve()

        return shift_vars.value if shift_vars.value is not None else np.zeros((self.num_employees, self.num_shifts))

    def generate_schedule(self, start_date, days):
        schedule = []
        current_date = start_date

        while len(schedule) < days * self.num_shifts:
            availability_predictions = self.predict_availability(current_date)

            shift_allocation = self.solve_optimization_problem(availability_predictions)

            schedule_entries = self.get_schedule_entries(shift_allocation, current_date)
            schedule.extend(schedule_entries)

            current_date += timedelta(days=1)

        return schedule

    def get_schedule_entries(self, shift_allocation, date):
        schedule_entries = []
        for i, employee_id in enumerate(self.employee_ids):
            for j in range(self.num_shifts):
                if shift_allocation[i, j] == 1:
                    schedule_entries.append({
                        'date': date,
                        'employee_id': employee_id,
                        'employee_name': employee_names[employee_id],
                        'shift_id': j
                    })
        return schedule_entries

    def get_shift_counts(self, schedule):
        df_schedule = pd.DataFrame(schedule)
        shift_counts = df_schedule.groupby('employee_name')['shift_id'].count().reset_index()
        shift_counts.columns = ['Employee Name', 'Shift Count']
        return shift_counts

    def get_shift_distribution(self, schedule):
        df_schedule = pd.DataFrame(schedule)
        shift_distribution = df_schedule['employee_name'].value_counts().reset_index()
        shift_distribution.columns = ['Employee Name', 'Shifts Worked']
        return shift_distribution

    def calculate_employee_stats(self, schedule, employee_name=None):
        df_schedule = pd.DataFrame(schedule)
        if employee_name:
            df_schedule = df_schedule[df_schedule['employee_name'] == employee_name]

        total_shifts = len(df_schedule)
        total_days = len(pd.date_range(start=min(df_schedule['date']), end=max(df_schedule['date']), freq='D'))
        off_days = total_days - len(df_schedule['date'].unique())

        total_hours = total_shifts * 8  # Assuming each shift is 8 hours

        return total_shifts, total_hours, off_days

# Function to convert DataFrame to CSV bytes for download
def dataframe_to_csv_bytes(df):
    csv = df.to_csv(index=False)
    csv = csv.encode()
    return csv

# Employee details
employee_ids = [1, 2, 3, 4, 5, 6, 7, 8]

optimizer = ScheduleOptimizer(employee_ids, num_shifts)

# Training the availability model with example data
warnings.filterwarnings("ignore", category=UserWarning)
X_train, y_train = make_classification(n_samples=100, n_features=10, n_informative=4, n_classes=num_shifts, random_state=42)
optimizer.train_availability_model(X_train, y_train)

# Initialize Streamlit app
st.set_page_config(page_title="Employee Scheduling App", page_icon=":clipboard:", layout='wide')
st.title("Automated Employee Scheduling Application")

# Custom CSS for a colorful theme
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #ff4d4d; /* Glossy Red */
    }
    .css-1l02zno {
        color: #333333;
        background-color: #4b0082; /* Dark Blue Purple */
    }
    .css-1u5g9rw {
        color: #ffffff;
        background-color: #ff4d4d; /* Glossy Red */
    }
    .st-bp {
        color: #5f6368;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar inputs and main logic
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Home", "Generate Schedule", "Employee Stats", "Dashboard"])

if page == "Home":
    st.write("Welcome to the Automated Employee Scheduling Application!")

elif page == "Generate Schedule":
    st.title("Generate Schedule")
    st.sidebar.header("Schedule Settings")
    num_days = st.sidebar.selectbox("Select Schedule Duration", [7, 14, 30], index=0)
    start_date = st.sidebar.date_input("Select Start Date", datetime.now().date())
    generate_button = st.sidebar.button("Generate Schedule")

    if generate_button:
        schedule = optimizer.generate_schedule(start_date, num_days)
        schedule_df = pd.DataFrame(schedule)
        st.write("Generated Schedule")
        st.dataframe(schedule_df)

        # Option to download generated schedule as CSV
        csv_bytes = dataframe_to_csv_bytes(schedule_df)
        st.download_button(label="Download Schedule as CSV", data=csv_bytes,
                           file_name=f"generated_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")

elif page == "Employee Stats":
    st.title("Employee Stats")
    st.sidebar.header("Employee Stats")

    # Option to upload CSV file for employee stats visualization
    uploaded_file = st.file_uploader("Upload a CSV file for employee stats visualization", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded DataFrame")
        
        # Filtering by employee names
        employee_names_list = list(employee_names.values())
        selected_employee = st.selectbox("Select Employee", ["All"] + employee_names_list)

        if selected_employee != "All":
            df = df[df['employee_name'] == selected_employee]

        st.dataframe(df)

        # Calculate and display shift statistics for uploaded data
        shift_counts = optimizer.get_shift_counts(df)
        st.write("Shifts Worked by Each Employee")
        st.dataframe(shift_counts)

        total_shifts, total_hours, off_days = optimizer.calculate_employee_stats(df)
        st.write(f"Total Shifts: {total_shifts}")
        st.write(f"Total Hours Worked: {total_hours}")
        st.write(f"Total Days Off: {off_days}")

elif page == "Dashboard":
    st.title("Dashboard")

    # Generate schedule for dashboard statistics
    num_days_dashboard = 30  # Adjust as needed for statistics calculation
    start_date_dashboard = datetime.now().date()  # Adjust start date as needed
    schedule_dashboard = optimizer.generate_schedule(start_date_dashboard, num_days_dashboard)

    # Employee shift distribution visualization
    shift_distribution = optimizer.get_shift_distribution(schedule_dashboard)
    fig = px.bar(shift_distribution, x='Employee Name', y='Shifts Worked', title='Employee Shift Distribution')
    st.plotly_chart(fig)

    # Summary statistics
    st.header("Summary Statistics")
    total_shifts, total_hours, off_days = optimizer.calculate_employee_stats(schedule_dashboard)
    st.write(f"Total Shifts Scheduled: {total_shifts}")
    st.write(f"Total Hours Worked: {total_hours}")
    st.write(f"Total Days Off: {off_days}")

    # Option to filter by employee for meter visualization
    employee_names_list = list(employee_names.values())
    selected_employee_dashboard = st.selectbox("Select Employee for Detailed Stats", ["All"] + employee_names_list, index=0)

    if selected_employee_dashboard != "All":
        total_shifts_employee, total_hours_employee, off_days_employee = optimizer.calculate_employee_stats(schedule_dashboard, selected_employee_dashboard)
        st.write(f"Total Shifts for {selected_employee_dashboard}: {total_shifts_employee}")
        st.write(f"Total Hours Worked by {selected_employee_dashboard}: {total_hours_employee}")
        st.write(f"Total Days Off for {selected_employee_dashboard}: {off_days_employee}")

        # Visualizations
        st.header(f"Metrics for {selected_employee_dashboard}")

        # Gauge for Total Shifts
        fig_shifts_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_shifts_employee,
            title={'text': "Total Shifts"},
            gauge={'axis': {'range': [0, max(shift_distribution['Shifts Worked'])]}}
        ))
        st.plotly_chart(fig_shifts_gauge)

        # Gauge for Total Hours Worked
        fig_hours_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_hours_employee,
            title={'text': "Total Hours Worked"},
            gauge={'axis': {'range': [0, max(shift_distribution['Shifts Worked'])*8]}}  # Assuming max hours is 8 times max shifts
        ))
        st.plotly_chart(fig_hours_gauge)

        # Gauge for Total Days Off
        fig_days_off_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=off_days_employee,
            title={'text': "Total Days Off"},
            gauge={'axis': {'range': [0, num_days_dashboard]}}
        ))
        st.plotly_chart(fig_days_off_gauge)
