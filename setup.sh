mkdir -p ~/.streamlit/





echo "\

Scratches and

[server]\n\

port = $PORT\n\

enableCORS = false\n\

headless = true\n\

\n\

"> ~/.streamlit/config.toml

web: sh setup.sh && streamlit run app.py