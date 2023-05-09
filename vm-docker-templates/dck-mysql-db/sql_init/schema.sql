USE database_name;

/*-----------------------------------------------------------------------*/
/*------------------------- A placeholder table -------------------------*/

CREATE TABLE IF NOT EXISTS placeholder_table (
    uid VARCHAR(16) NOT NULL,
    description VARCHAR(255),
    PRIMARY KEY (uid)
);
