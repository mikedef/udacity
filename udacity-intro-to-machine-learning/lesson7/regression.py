def studentReg(ages_train, net_worths_train):
    ### import the sklearn regression module, create, and train your regression
    ### name your regression reg
    
    ### your code goes here!
    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit (ages_train, net_worths_train)

    print "Net worth prediction:", reg.predict([27]) # must be in the form of a list even though there is only one value
    print "slope", reg.coef_
    print "intercept:", reg.intercept_
    print "\n ####### stats on test data ########\n"
    print "r-squared score", reg.score(ages_test, net_worths_test)
    print "\n ###### stats on training dataset ########\n"
    print "r-squared score", reg.score(ages_train, net_worth_train)
    
    
    return reg
