FROM python:3.11.11-bookworm

WORKDIR /env

COPY requirements.txt .
COPY experimental_code/*py .

RUN apt-get update && \
    apt-get install --no-install-recommends -y latexmk texlive-full
RUN pip install latex
RUN pip install -r requirements.txt

CMD ["python", "chp3-the_sbi_package.py", "&&", "cp", "-r", "figs" "/mnt"]
CMD python chp3-the_sbi_package.py && cp -r figs /mnt
