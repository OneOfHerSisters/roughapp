# Frontend image: build the Vite app, then serve the static bundle via nginx.
FROM node:20-alpine AS build

WORKDIR /app

# Install dependencies first for layer caching.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ── serve ──────────────────────────────────────────────────────────────────
FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
