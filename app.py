import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


from plotly import graph_objs as go

def home(data):
    if st.checkbox("Show table:"):
        st.table(data)
    graph=st.selectbox("Select type of graph:",['Interactive','Non-Interactive'])
    if graph=='Non-Interactive':
        plt.figure(figsize=(10,5))
        plt.scatter(data['YearsExperience'],data['Salary'])
        plt.ylim(0)
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.tight_layout()
        st.pyplot()
    if graph=='Interactive':
        layout =go.Layout(
            xaxis = dict(range=[0,16]),
            yaxis = dict(range =[0,210000])
        )
        fig = go.Figure(data=go.Scatter(x=data["YearsExperience"], y=data["Salary"], mode='markers'),layout = layout)
        st.plotly_chart(fig)

def model_train(data):
    x=np.array(data['YearsExperience']).reshape(-1,1)
    lr=LinearRegression()
    lr.fit(x,np.array(data['Salary']))
    return lr
def predict_value(model,value):
    pred=model.predict(value)[0]
    return pred

    
def contri_to_data(exp,salary):
    data_add={'YearsExperience':[exp],'Salary':[salary]}
    df_add=pd.DataFrame(data_add)
    df_add.to_csv('Contri_Salary_data.csv',mode='a',index=False,header=False)
    
    



def main():
    data=pd.read_csv('Salary_Data.csv')
    lr=model_train(data)
    rad=st.sidebar.radio("Navigation:",['Home','Prediction','Contribute'])
    if rad=='Home':
        home(data)

    if rad=='Prediction':
        st.header('Know your Salary')
        val=st.number_input("Enter your salary",0.0,20.00,step=0.25)
        if st.button("Predict"):
            val=np.array(val).reshape(1,-1)
            prediction=predict_value(lr,val)
            st.success(f'Your predicted Salary is {round(prediction)}')
         
    if rad=='Contribute':
        st.header('Contribute to data')
        exp=st.number_input("Enter you Experience:",0.0,20.0)
        salary=st.number_input("Enter your salary:",0.0,100000.0,step=1000.0)
        if st.button('Submit'):
            contri_to_data(exp,salary)
            st.success('Thanks for contribuing')
if __name__=='__main__':
    main()