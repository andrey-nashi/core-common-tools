# ---------------------------------------------------------
# Docker for a MYSQL database, nothing else
# ---- * sql_init/schema.sql - an sql script to create DB tables
# ---- * sql_config/my.cnf - default mysql config file
# ---------------------------------------------------------
FROM mysql:8.0

# >>>> FIXME set the correct credentials here
ENV MYSQL_DATABASE=lesser_dicom_db \
  MYSQL_USER=db_user \
  MYSQL_PASSWORD=db_pass_1234$ \
  MYSQL_ROOT_PASSWORD=db_pass_1234$ \
  TZ=Asia/Tokyo

# >>>> FIXME this is the default port, use other if needed
EXPOSE 3306
