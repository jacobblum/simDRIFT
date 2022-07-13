import smtplib


USER = r"jacob.sam.blum@gmail.com"
PASS = r"@Wildtwins23"
data = "TEST 123"
s = smtplib.SMTP_SSL('smtp.mail.com',465)
s.login(USER, PASS)
s.sendmail(USER, USER,data)
s.quit()