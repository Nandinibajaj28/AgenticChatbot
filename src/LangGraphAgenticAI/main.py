import streamlit as st
from src.LangGraphAgenticAI.ui.streamlitUI.loadui import LoadStreamlitUI
from src.LangGraphAgenticAI.ui.streamlitUI.display_result import DisplayResultStreamlit
from src.LangGraphAgenticAI.graph.graph_builder import GraphBuilder
from src.LangGraphAgenticAI.LLMs.groqllm import GroqLLM




def load_langgraph_agenticai_app():
    ##Load UI
    ui=LoadStreamlitUI() #ui= class in loadui.py
    user_input=ui.load_streamlit_ui() #user_input= function in loadui.py
    
    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
        return
    
    # Text input for user message
    if st.session_state.IsFetchButtonClicked:
        user_message = st.session_state.timeframe 
    else :
        user_message = st.chat_input("Enter your message:")
    
    if user_message:
        try:
            ## Configure The LLM's
            obj_llm_config=GroqLLM(user_contols_input=user_input)
            model=obj_llm_config.get_llm_model()

            if not model:
                st.error("Error: LLM model could not be initialized")
                return
            
            # Initialize and set up the graph based on use case
            usecase=user_input.get("selected_usecase")

            if not usecase:
                st.error("Error: No use case selected.")
                return
            
            ## Graph Builder

            graph_builder=GraphBuilder(model)
            try:
                graph=graph_builder.setup_graph(usecase)
                print(user_message)
                DisplayResultStreamlit(usecase,graph,user_message).display_result_on_ui()
            except Exception as e:
                st.error(f"Error: Graph set up failed- {e}")
                return

        except Exception as e:
            st.error(f"Error: Graph set up failed- {e}")
            return


