def retry(how_many_tries = 2):
   def wrapper(func):
     def try_it(*args,**kwargs):
       tries = 0
       while tries < how_many_tries:
         try:
           return func(*args,**kwargs)
         except:
           tries +=1
       return -1
     return try_it
   return wrapper
 

