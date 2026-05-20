from setuptools import setup, find_packages
 
setup(
    name="multi-agent-system",
    version="0.1.0",
    author="Pulkit Kushwaha",
    author_email="pulkitkushwahadev@gmail.com",
    description="Multi-agent AI system built with LangGraph — planner, retriever, and synthesizer agents with evaluation harness and failure mode documentation",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "langchain>=0.1.20",
        "langgraph>=0.0.38",
        "openai>=1.14.0",
        "pydantic>=2.6.4",
        "faiss-cpu>=1.7.4",
    ],
)
