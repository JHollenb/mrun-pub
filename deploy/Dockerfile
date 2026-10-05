FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir '.[server]'
RUN useradd --uid 10001 --create-home mrun && mkdir /data && chown mrun /data
USER mrun
ENV MRUN_BIND=0.0.0.0 MRUN_PORT=9025 MRUN_SERVER_DATA=/data
EXPOSE 9025
CMD ["python", "-m", "mrun.server"]
