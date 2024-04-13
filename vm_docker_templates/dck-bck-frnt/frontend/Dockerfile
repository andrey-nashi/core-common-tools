FROM node:20.9

# set working directory
WORKDIR /app

ENV PATH /app/node_modules/.bin:$PATH

# install app dependencies
COPY ./frontend ./
RUN npm install --silent

#CMD ["npm", "start"]