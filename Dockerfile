FROM public.ecr.aws/lambda/python:3.8
RUN pip install numpy
RUN pip install pandas
RUN pip install xgboost
RUN pip install scikit-learn
RUN pip install requests
COPY model_fraud.model .
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]