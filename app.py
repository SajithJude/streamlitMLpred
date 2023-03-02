
import streamlit as st

# st.sidebar.success("Select a demo above.")

st.markdown("# Bipolar Disorder Analytical Platform")
st.markdown("## Made by Researches for Researchers for the betterment of cognitive phychology")
st.markdown("""
    
#### Click on the arrow on the left to open Menue, 
#### Input the URLS of the publications that revolve around your domain,
#### Start with the link that is the most relevant
#### Once you click on the generate button click on the titles of the generated content to expand
#### Repeat the process with atleast 4 URLS
#### Once youre Finished, click on the Review table to access all the generated content together

  This app is designed to help researchers analyze their Twitter histories and predict the likelihood of developing bipolar disorder. By using advanced algorithms and machine learning techniques, our app can analyze thousands of tweets and identify patterns and trends that may indicate the presence of bipolar disorder.
The app works by analyzing a user's Twitter history and looking for key phrases and language that are commonly associated with bipolar disorder. It takes into account factors such as the frequency and intensity of mood swings, the use of specific words or phrases that may indicate manic or depressive episodes, and other indicators of mental health status.
Users can simply upload their Twitter data to the app and allow it to analyze their tweets. The app will then provide a comprehensive report detailing the user's risk of developing bipolar disorder based on their Twitter activity. This report can be used to inform users about their mental health status and encourage them to seek professional help if necessary.
Our app is designed to be easy to use and accessible to everyone. It is a powerful tool for anyone who is concerned about their mental health and wants to take proactive steps to prevent the onset of bipolar disorder. With our app, users can take control of their mental health and stay on top of their mental wellness.
  
  """)

with st.container():
    image_col, text_col = st.columns((1,2))
    with image_col:
        st.image("https://media.licdn.com/dms/image/C4D03AQEK4oRhXVTLKA/profile-displayphoto-shrink_800_800/0/1656655465659?e=2147483647&v=beta&t=ePOmciJOCtT_JvhLwafdgf1UMC7f5tL-uRMdBSJlVGI")

    with text_col:
        st.subheader("Developed by Nisadi Jayasuriya")
        # st.write("""Connect with me on LinkedIn for quick replies
            # """)
        st.markdown("[LinkedIn profile...]https://lk.linkedin.com/in/nisandi-jayasuriya-294327194?trk=people_directory&original_referer=)")
