def load_professional_data():
    """Load professional experience data"""
    professional_data = {
        "experience": [
            {
                "role": "Analytics Engineer",
                "company": "Sitio Royalties",
                "duration": "March 2025 - Present",
                "description": "Built Databricks-native Streamlit Acquisitions Dashboard to centralize scenario analysis and streamline ingestion of deal-specific data. Designed Type Curve Evaluation Tool in Spotfire for Reservoir Engineers to compare type curves to actual production data. Led Databricks Catalog governance initiative and partnered cross-functionally with Data Engineers and Business Development.",
                "skills": ["Databricks", "Streamlit", "Spotfire", "Python", "CSV/Excel Processing", "Data Governance", "Cross-functional Collaboration"]
            },
            {
                "role": "Advanced Analytics Business Intelligence Analyst",
                "company": "ConocoPhillips",
                "duration": "July 2022 - March 2025",
                "description": "Designed and implemented agnostic ChatBot UI class using Streamlit with Snowflake backend, enabling interaction with various Large Language Models including LLAMA and OpenAI. Produced User Interface for administering LLMs and engineered Classification Relabeling UI. Developed Predictive Decline Model Spotfire reports connected to AWS housed models. Re-architected Spotfire IronPython scripts reducing AWS payload by 98%.",
                "skills": ["Streamlit", "Snowflake", "Python", "Large Language Models", "OpenAI", "LLAMA", "Spotfire", "AWS", "SQL Server", "Power BI", "DAX", "Power Automate", "Agile Methodology"]
            },
            {
                "role": "Data Engineer",
                "company": "ConocoPhillips",
                "duration": "December 2021 - July 2022",
                "description": "Developed data pipelines in AWS using Lambda Functions, Gitlab, and Python for 2 different Machine Learning Models. Upgraded in-house Model Server solution on AWS by integrating advanced logging and monitoring features. Designed and implemented Streamlit application hosted on AWS to track usage and health of internal Model Server solution.",
                "skills": ["AWS", "Lambda Functions", "Python", "GitLab", "Machine Learning", "Streamlit", "Model Server", "Logging", "Monitoring"]
            },
            {
                "role": "Corporate Staff Business Intelligence Analyst",
                "company": "ConocoPhillips",
                "duration": "December 2020 - December 2021",
                "description": "Managed analytics strategies including Power BI gateways, reports/dashboards, and long-term analytic objectives for 6 different business teams. Created Legal Financial Power BI report providing 3 hours weekly time-savings. Instituted Power BI solution for United Way campaigns saving 40 hours quarterly.",
                "skills": ["Power BI", "Analytics Strategy", "Dashboard Development", "Data Visualization", "Business Intelligence", "Process Automation"]
            },
            {
                "role": "Solution Integrator",
                "company": "ConocoPhillips",
                "duration": "May 2019 - December 2020",
                "description": "Integrated new HSE data management system into Teradata Data Warehouse using SQL. Overhauled HSE dashboards in Spotfire to merge new system with historical data. Developed Environmental and Regulatory Dashboards by standardizing data formats from multiple business units.",
                "skills": ["SQL", "Teradata", "Spotfire", "Data Integration", "Dashboard Development", "Data Standardization"]
            },
            {
                "role": "Analytics Analyst Intern",
                "company": "ConocoPhillips",
                "duration": "May 2018 - August 2018",
                "description": "Created sentiment analysis tool using Python Natural Language Toolkit to monitor news outlets for news pertaining to ConocoPhillips.",
                "skills": ["Python", "Natural Language Processing", "NLTK", "Sentiment Analysis", "News Monitoring"]
            }
        ],
        "projects": [
            {
                "name": "Databricks-native Streamlit Acquisitions Dashboard",
                "description": "Built comprehensive acquisitions dashboard using Databricks Lakehouse architecture to centralize scenario analysis and streamline ingestion of deal-specific CSV, Excel, and shapefile data tied to acquisition opportunities.",
                "tech_stack": ["Databricks", "Streamlit", "Python", "CSV Processing", "Excel Processing", "Shapefile Processing"],
                "impact": "Centralized scenario analysis and streamlined data ingestion for acquisition opportunities"
            },
            {
                "name": "Type Curve Evaluation Tool",
                "description": "Designed and developed a comprehensive evaluation tool in Spotfire that enabled Reservoir Engineers to compare type curves to actual production data across assets, identify under- or over-performing wells, and iteratively generate improved type curves.",
                "tech_stack": ["Spotfire", "Python", "Production Data Analysis", "Statistical Modeling"],
                "impact": "Enabled engineers to identify performance gaps and generate improved type curves based on historical performance"
            },
            {
                "name": "Multi-Model ChatBot UI Platform",
                "description": "Developed agnostic ChatBot UI class using Streamlit with Snowflake backend, enabling interaction with various Large Language Models including LLAMA and OpenAI models. Facilitated user interaction with 5 different chatbots trained on internal data sources.",
                "tech_stack": ["Streamlit", "Snowflake", "Python", "OpenAI", "LLAMA", "Large Language Models"],
                "impact": "Enabled seamless interaction with multiple AI models trained on internal data sources"
            },
            {
                "name": "Predictive Decline Model Integration",
                "description": "Developed Spotfire reports for 3 assets connected to AWS housed models to allow Drilling Engineers the ability to predict Oil, Gas, and Water production for current and future projects. Optimized IronPython scripts reducing AWS payload by 98%.",
                "tech_stack": ["Spotfire", "AWS", "IronPython", "Machine Learning Models", "Production Forecasting"],
                "impact": "Reduced AWS payload by 98% and enabled accurate production predictions for drilling projects"
            }
        ],
        "skills": {
            "business_intelligence": ["Power BI", "Tableau", "Spotfire", "Streamlit", "Excel"],
            "programming": ["Python", "SQL", "DAX", "Spark"],
            "data_platforms": ["Databricks", "Snowflake", "SQL Server", "Teradata"],
            "cloud_platforms": ["AWS", "Azure"],
            "ml_frameworks": ["Large Language Models", "OpenAI", "LLAMA", "Natural Language Processing", "NLTK"],
            "engineering_tools": ["Jira", "GitLab", "GitHub", "Confluence", "Domino Data Labs"],
            "soft_skills": ["Effective Communication", "Collaboration", "Adaptability", "Mentoring", "Cross-functional Partnership"]
        },
        "certifications": [
            {
                "name": "PL-300: Microsoft Power BI Data Analyst",
                "year": "Not specified",
                "description": "Microsoft certification in Power BI data analysis and visualization"
            }
        ],
        "education": {
            "degree": "Bachelor of Science in Computer Science",
            "university": "University of Arkansas - Fayetteville",
            "year": "2019",
            "minor": "Data Analytics",
            "relevant_coursework": ["Computer Science", "Data Analytics"]
        }
    }
    return professional_data