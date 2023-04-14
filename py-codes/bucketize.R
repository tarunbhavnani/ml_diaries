
dat<-data.frame(n=seq(1:1000),age=sample(27:90,1000,T), salary=sample(1000:10000,1000,T)
                 , churn=sample(c(0,1),1000,T))

#the input has to be in this format:dataset[c("variable_to_be_operated","churn")], the churn variable should be named churn 


require(dplyr)
bucketize<-function(a){

a<-as.data.frame(a)
placeholder=names(a)[1]
placeholder_churn=names(a)[2]
a$bracket<-ifelse(a[,1]<as.numeric(paste0(floor(quantile(a[,1],.1)))),
                  paste0("<",paste0(floor(quantile(a[,1],.1)))),
                   ifelse(a[,1]<as.numeric(paste0(floor(quantile(a[,1],.2)))) & a[,1]>=as.numeric(paste0(floor(quantile(a[,1],.1)))),
                    paste0(paste0(floor(quantile(a[,1],.1))),"-",paste0(floor(quantile(a[,1],.2)))),
                     ifelse(a[,1]<as.numeric(paste0(floor(quantile(a[,1],.3)))) & a[,1]>=as.numeric(paste0(floor(quantile(a[,1],.2)))),
                       paste0(paste0(floor(quantile(a[,1],.2))),"-",paste0(floor(quantile(a[,1],.3)))),
                        ifelse(a[,1]<as.numeric(paste0(floor(quantile(a[,1],.4)))) & a[,1]>=as.numeric(paste0(floor(quantile(a[,1],.3)))),
                          paste0(paste0(floor(quantile(a[,1],.3))),"-",paste0(floor(quantile(a[,1],.4)))),
                            ifelse(a[,1]<as.numeric(paste0(floor(quantile(a[,1],.5)))) & a[,1]>=as.numeric(paste0(floor(quantile(a[,1],.4)))),
                             paste0(paste0(floor(quantile(a[,1],.4))),"-",paste0(floor(quantile(a[,1],.5)))),
                              ifelse(a[,1]<as.numeric(paste0(floor(quantile(a[,1],.6)))) & a[,1]>=as.numeric(paste0(floor(quantile(a[,1],.5)))),
                               paste0(paste0(floor(quantile(a[,1],.5))),"-",paste0(floor(quantile(a[,1],.6)))),
                                ifelse(a[,1]<as.numeric(paste0(floor(quantile(a[,1],.7)))) & a[,1]>=as.numeric(paste0(floor(quantile(a[,1],.6)))),
                                 paste0(paste0(floor(quantile(a[,1],.6))),"-",paste0(floor(quantile(a[,1],.7)))),
                                  ifelse(a[,1]<as.numeric(paste0(floor(quantile(a[,1],.8)))) & a[,1]>=as.numeric(paste0(floor(quantile(a[,1],.7)))),
                                   paste0(paste0(floor(quantile(a[,1],.7))),"-",paste0(floor(quantile(a[,1],.8)))),
                                    ifelse(a[,1]<as.numeric(paste0(floor(quantile(a[,1],.9)))) & a[,1]>=as.numeric(paste0(floor(quantile(a[,1],.8)))),
                                     paste0(paste0(floor(quantile(a[,1],.8))),"-",paste0(floor(quantile(a[,1],.9)))),
                                      paste0(paste0(">=",paste0(floor(quantile(a[,1],.9))))))))))))))

mx<-eval(parse(text=paste0("aggregate(",placeholder,"~bracket,data=a, max)"))); names(mx)<-c("bracket",paste0("max_",names(mx)[2]))
min<-eval(parse(text=paste0("aggregate(",placeholder,"~bracket, data=a,function(x)min(x))")));names(min)<-c("bracket",paste0("min_",names(min)[2]))
count<-eval(parse(text=paste0("aggregate(",placeholder,"~bracket,data=a, length)"))); names(count)<-c("bracket","count")
mean<-eval(parse(text=paste0("aggregate(",placeholder,"~bracket,data=a,function(x) mean(x))"))); names(mean)<-c("bracket",paste0("mean_",names(mean)[2]))
churn<-eval(parse(text=paste0("aggregate(",placeholder_churn,"~bracket,data=a, sum)"))); names(churn)<-c("bracket",paste0("Total_",names(churn)[2]))
a<-merge(a,mx,by="bracket")%>%merge(min,by="bracket")%>%merge(count,by="bracket")%>%merge(mean,by="bracket")%>%merge(churn,by="bracket")
a$bracket<-as.factor(a$bracket)
c<-aggregate(.~bracket, data=a[,-2], function(x)mean(x))
c$Churn_pc<-(c$Total_churn/sum(c$Total_churn))*100
c$Non_Churn_pc<-((c$count-c$Total_churn)/(sum(c$count)-sum(c$Total_churn)))*100
c$pc_diff<-c$Non_Churn_pc-c$Churn_pc
c$odds<-c$Non_Churn_pc/c$Churn_pc
c$ln_odds<-log(c$odds)
c<-c[order(c[names(c)[grep("min_", names(c))]]),]
return(c)
}

sal<-bucketize(dat[c("salary","churn")])
age<-bucketize(dat[c("age","churn")])
names(dat)





