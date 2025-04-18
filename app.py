import streamlit as st
from langchian.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagePlaceHolder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory


from dotenv import load_dotenv

load_dotenv()

os['HF_TOKEN']=os.getenv('HF_TOKEN')
embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')


##set up the streamlit app
st.title('Conversational RAG with PDF Uploads and Chat History')
st.write('upload the pdf and chat with the content')

##input the Groq API KEY
api_key=st.text_input('Enter the GROQ API KEY',type='password')

##check if groq api key is provided

if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name='Gemma2-9b-It')


    ###chat interface
    session_id=st.text_input('session_id',value='default session')

    ##statefully manage chat history
    
    if 'store' not in st.session_state:
        st.session_store={}
        
    uploaded_files=st.file_uploader('Choose A PDF file',type='pdf',accept_multiple_files=True)

    ##Process uploaded a file
    if uploaded_file:
        documents=[]
        for uploaded_file in uploaded_files:
            temp_pdf=f'./temp.pdf'
            with open(temp_pdf,'wb') as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temp_pdf)
            docs=loader.load()
            documents.extend(docs)

        ##split and create the embedding for the documents
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vectorstoredb=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vectorstore.as_retriever()

        contextualize_q_system_prompt=(
                'Given a chat history and the latest user question'
                'Which might reference context from the Chat History'
                'formulate a standalone question which can be understood'
                'without the chat history do not answer the question'
                'just reformulate it if needed and otherwise return as it is'
        )

        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ('system',contextualize_q_system_prompt),
                MessagePlaceholder('chat_history'),
                ('human','{input}')
            ]
        )

        ##creating history aware retriever
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        ##Answer Question Prompt
        system_prompt= (
            'you are an Assistant for question-answering tasks,'
            'Use the following pieces of retrieved context to answer'
            'the question. If you dont know the answer,say that you '
            'dont know. Use three sentences maximum and keep the '
            'answer consise'
            '\n\n'
            '{context}'
        )

        #qa prompt
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ('system',system_prompt),
                MessagePlaceHolder('chat_history'),
                ('human','{input}')
            ]
        )

        question_answer_chain=create_stuff_document_chain(llm,qa_prompt)

        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
        
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_state]=ChatMessageHistory()
            
            return st.session_state.store[session_id]

        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            ooutput_messages_key='answer'
        )

        user_input=st.text_input('Your Question')

        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {
                    'input':'{user_input}'
                },
                config={
                    'configurable':{'session_id':session_id}
                },
            )
            st.write(st.session_state.store)
            st.write('Assistant',response['answer'])
            st.write('Chat History',session_history.messages)

else:
    st.write('please enter your API Key')

        



