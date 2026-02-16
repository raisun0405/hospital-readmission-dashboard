"""
Data Visualization Script
Creates visualizations for the dashboard
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

def create_data_visualizations(df_path='data/diabetic_data.csv', output_dir='models'):
    """Create visualizations from raw data"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("📊 Loading data for visualizations...")
    df = pd.read_csv(df_path)
    
    # Create readmission target
    df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
    
    # 1. Readmission by Age Group
    fig1 = px.histogram(df, x='age', color='readmitted',
                       title='Readmission Distribution by Age Group',
                       labels={'age': 'Age Group', 'count': 'Number of Patients'},
                       barmode='group')
    fig1.write_html(f'{output_dir}/viz_age_readmission.html')
    fig1.write_image(f'{output_dir}/viz_age_readmission.png', width=1000, height=600)
    
    # 2. Readmission by Time in Hospital
    fig2 = px.box(df, x='readmitted', y='time_in_hospital',
                  title='Length of Stay vs Readmission',
                  labels={'time_in_hospital': 'Days in Hospital', 'readmitted': 'Readmission Status'})
    fig2.write_html(f'{output_dir}/viz_stay_readmission.html')
    fig2.write_image(f'{output_dir}/viz_stay_readmission.png', width=1000, height=600)
    
    # 3. Number of Medications vs Readmission
    med_readmit = df.groupby(['num_medications', 'readmitted']).size().reset_index(name='count')
    fig3 = px.scatter(med_readmit, x='num_medications', y='count', color='readmitted',
                     title='Number of Medications vs Readmission',
                     size='count')
    fig3.write_html(f'{output_dir}/viz_medications_readmission.html')
    fig3.write_image(f'{output_dir}/viz_medications_readmission.png', width=1000, height=600)
    
    # 4. Gender and Race distribution
    fig4 = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Readmission by Gender', 'Readmission by Race'))
    
    gender_data = df.groupby(['gender', 'readmitted']).size().unstack()
    for col in gender_data.columns:
        fig4.add_trace(go.Bar(x=gender_data.index, y=gender_data[col], name=str(col)), row=1, col=1)
    
    race_data = df.groupby(['race', 'readmitted']).size().unstack()
    for col in race_data.columns:
        fig4.add_trace(go.Bar(x=race_data.index, y=race_data[col], name=str(col), showlegend=False), row=1, col=2)
    
    fig4.update_layout(title_text="Readmission by Demographics", barmode='group')
    fig4.write_html(f'{output_dir}/viz_demographics.html')
    fig4.write_image(f'{output_dir}/viz_demographics.png', width=1200, height=500)
    
    # 5. Summary statistics
    stats = {
        'total_patients': len(df),
        'readmitted_30': int((df['readmitted'] == '<30').sum()),
        'readmitted_30_pct': round((df['readmitted'] == '<30').mean() * 100, 2),
        'readmitted_gt30': int((df['readmitted'] == '>30').sum()),
        'not_readmitted': int((df['readmitted'] == 'NO').sum()),
        'avg_age': df['age'].mode()[0] if not df['age'].mode().empty else 'N/A',
        'avg_stay': round(df['time_in_hospital'].mean(), 2),
        'avg_medications': round(df['num_medications'].mean(), 2),
        'avg_diagnoses': round(df['number_diagnoses'].mean(), 2)
    }
    
    with open(f'{output_dir}/dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Visualizations saved to {output_dir}/")
    print(f"📊 Dataset stats:")
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    return stats

if __name__ == "__main__":
    stats = create_data_visualizations()
    print("\n🎉 Visualizations complete!")
