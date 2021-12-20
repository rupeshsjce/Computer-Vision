class Test(object):
      
    # This function prints the type
    # of the object passed as well 
    # as the object item
    def __getitem__(self, items):
        print (type(items), items)
        return (type(items), items)
  
# Driver code
test = Test()
print("outside ", test[5])