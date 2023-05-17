FROM ivy_pip_gradslam_img:latest 
WORKDIR /ivy/ivy
ENV API_KEY="API_KEY_VALUE"
RUN mkdir -p .ivy
RUN echo -n "API_KEY_VALUE" > .ivy/key.pem
# FROM unifyai/ivy:latest
# RUN pip install gradslam

# RUN git clone https://github.com/krrish94/chamferdist.git
# RUN cd chamferdist
# RUN pip install .
# RUN cd ..
# RUN git clone https://github.com/gradslam/gradslam.git
# RUN cd gradslam
# RUN pip install -e .[dev]
