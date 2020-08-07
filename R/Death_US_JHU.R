rm(list=ls(all=TRUE))

library(ggplot2)
library(dplyr)
library(gridExtra)
library(ggpubr)
library(scales)

get_CIdat <- function(dat, p)
{
  lower = apply(dat, 2, function(x) quantile(x, probs=p))
  lower_dat = data.frame(date = as.Date(firstday, '%m/%d/%y') + c(0:(length(lower)-1)), 
                         new_obs = lower)
}


library (readr)
urlfile="https://urldefense.proofpoint.com/v2/url?u=https-3A__raw.githubusercontent.com_CSSEGISandData_COVID-2D19_master_csse-5Fcovid-5F19-5Fdata_csse-5Fcovid-5F19-5Ftime-5Fseries_time-5Fseries-5Fcovid19-5Fdeaths-5FUS.csv&d=DwIFAg&c=G2MiLlal7SXE3PeSnG8W6_JBU6FcdVjSsBSbw6gcR0U&r=TvHTLzJxxNNJvKfeCGpokIVmFYhgBRj_ERlMwxdZbeQ&m=bT4oYCtgK4sEfCL2vpGuaWqQ2zMvqdKw8JyJiBOdXl4&s=nayi0ReGfecf5WOhxVcWT00t26bU1TyNO46a_Sp9Q0k&e= "
jhu<-read_csv(url(urlfile))
colnames(jhu)


## sum over all rows to get cumulative death of US
us_death0 = jhu[,c(13:length(jhu))] %>% apply(2, sum)

### US daily death
us_daily_death = c(0,diff(us_death0))

### correct a death peak at 6/25
us_daily_death[156] = 681
tmp_dif = 2422-681
us_death = cumsum(us_daily_death)


### read predicted daily case

dat = read.csv("all_obs_param30.csv")   ## our data file (format: first column: date; second column: new_obs observed daily case, third column: pred (label indicates Train or Predicted or Test))


dat$date = as.Date(dat$date)

firstday = dat$date[1]
pred=dat[dat$pred=="Predicted",c("date","new_obs","pred")]

cumcasepred=cumsum(pred$new_obs)

pred=cbind(pred,cumcasepred)


lastobsday=as.Date("2020-07-31")

pred_obs=pred%>%filter(date <=lalastobsday)
cumdeath=as.numeric(us_death)
deathdate=names(us_death)
deathdate=as.Date(deathdate,"%m/%d/%y")
cumdeath=cumdeath[31:length(us_death)] ## from use data starting from 2/21

pred_obs=cbind(pred_obs,cumdeath)
incdeath=c(0,diff(cumdeath))
pred_obs=cbind(pred_obs,incdeath)

#### create training data

pred_train=pred_obs%>%filter(date <=lastobsday)

colnames(pred_train)[5]="cumdeath"

### To predict log cumulative death using a linear regression of log predicted cumulative case one day before, 7 days before, and 14 days before with knots to account for reopen
#log(cumulative death at t)~log(cumulative predicted case at t-1)*reopen1+log(cumulative predicted case at t-7)*reopen1+
## log(cumulative predicted case at t-14)*reopen1+log(cumulative predicted case at t-1)*reopen2+log(cumulative predicted case at t-7)*reopen2+
## log(cumulative predicted case at t-14)*reopen2+log(cumulative predicted case at t-1)*reopen3+log(cumulative predicted case at t-7)*reopen3+
## log(cumulative predicted case at t-14)*reopen3

## 14 days before cumulative case
cumcase_pred_lag14=lag(pred_train$cumcasepred,14)  
## 1 days before cumulative case
cumcase_pred_lag1=lag(pred_train$cumcasepred,1)
## 7 days before cumulative case
cumcase_pred_lag7=lag(pred_train$cumcasepred,7)

pred_train=cbind(pred_train,cumcase_pred_lag21,cumcase_pred_lag14,cumcase_pred_lag1,cumcase_pred_lag7)


pred_train=na.omit(pred_train)
pred_train=pred_train%>%filter(date >="2020-03-26")  ### using data starting from 3/26 (for state levle, choose reasonable date when the number of death is not too small)

### log of cumulative case
log_cumcase_pred_lag1=log(pred_train$cumcase_pred_lag1)
log_cumcase_pred_lag7=log(pred_train$cumcase_pred_lag7)
log_cumcase_pred_lag14=log(pred_train$cumcase_pred_lag14)
log_cumcase_pred_lag21=log(pred_train$cumcase_pred_lag21)

pred_train=cbind(pred_train,log_cumcase_pred_lag21,log_cumcase_pred_lag14,log_cumcase_pred_lag1,log_cumcase_pred_lag7)

log_cumdeath=log(pred_train$cumdeath)

pred_train=cbind(pred_train,log_cumdeath)

pred_train=pred_train%>%filter(date <=lastobsday)

reopen=ifelse(pred_train$date>="2020-05-1",1,0)
pred_train=cbind(pred_train,reopen)

reopen2=ifelse(pred_train$date>="2020-05-22",1,0)
pred_train=cbind(pred_train,reopen2)

reopen3=ifelse(pred_train$date>="2020-06-26",1,0)
pred_train=cbind(pred_train,reopen3)

### fit regression on training data to predict log cumulative death

## add 6/26 knot
lmfit=lm(log_cumdeath~log_cumcase_pred_lag1+log_cumcase_pred_lag7+log_cumcase_pred_lag14+reopen+reopen:log_cumcase_pred_lag1+reopen:log_cumcase_pred_lag7+reopen:log_cumcase_pred_lag14+reopen2+reopen2:log_cumcase_pred_lag1+reopen2:log_cumcase_pred_lag7+reopen2:log_cumcase_pred_lag14+reopen3+reopen3:log_cumcase_pred_lag1+reopen3:log_cumcase_pred_lag7+reopen3:log_cumcase_pred_lag14,data=pred_train)
summary(lmfit)

## use residual of the original scale
death_res = (exp(log_cumdeath)-exp(fitted(lmfit)))


## plot cumulative death residuals
plot(death_res,x=seq(as.Date("2020-03-26"),lastobsday,1),type="o",xaxt = "n",xlab="Date",main="cumulative death model")
axis(1, at=seq(as.Date("2020-03-26"),lastobsday,10), labels=seq(as.Date("2020-03-26"),lastobsday,10))


### do prediction
pred_test=pred

cumcase_pred_lag21=lag(pred_test$cumcasepred,21)
cumcase_pred_lag14=lag(pred_test$cumcasepred,14)
cumcase_pred_lag1=lag(pred_test$cumcasepred,1)
cumcase_pred_lag7=lag(pred_test$cumcasepred,7)

pred_test=cbind(pred_test,cumcase_pred_lag21,cumcase_pred_lag14,cumcase_pred_lag1,cumcase_pred_lag7)


pred_test=na.omit(pred_test)

pred_test=pred_test%>%filter(date >="2020-03-26")

log_cumcase_pred_lag1=log(pred_test$cumcase_pred_lag1)
log_cumcase_pred_lag7=log(pred_test$cumcase_pred_lag7)
log_cumcase_pred_lag14=log(pred_test$cumcase_pred_lag14)
log_cumcase_pred_lag21=log(pred_test$cumcase_pred_lag21)

pred_test=cbind(pred_test,log_cumcase_pred_lag21,log_cumcase_pred_lag14,log_cumcase_pred_lag1,log_cumcase_pred_lag7)

reopen=ifelse(pred_test$date>="2020-05-1",1,0)
pred_test=cbind(pred_test,reopen)

reopen2=ifelse(pred_test$date>="2020-05-22",1,0)
pred_test=cbind(pred_test,reopen2)

reopen3=ifelse(pred_test$date>="2020-06-26",1,0)
pred_test=cbind(pred_test,reopen3)

### predict
log_cumdeath_pred_test=predict(lmfit, newdata = pred_test)

cumdeath_pred_test=exp(log_cumdeath_pred_test)

pred_test=cbind(pred_test,cumdeath_pred_test,log_cumdeath_pred_test)

incdeath_pred_test=c(0,diff(cumdeath_pred_test))

pred_test=cbind(pred_test,incdeath_pred_test)

train_predicted=pred_test%>%filter(date <=lastobsday)

train_predicted=cbind(train_predicted,death_res)

sum(abs(pred_train$cumdeath-train_predicted$cumdeath_pred_test))/nrow(pred_train)  ## 265.0801 (knot 6/26)



cumdeath_dat=data.frame(date=pred_train$date,cumdeath=pred_train$cumdeath,pred="Train")
## adjust by adding tmp_diff back
cumdeath_dat = cumdeath_dat %>% mutate(cumdeath = ifelse(date>=as.Date('2020-06-25'), cumdeath+tmp_dif, cumdeath))


cumdeath_pred=data.frame(date=seq(as.Date("2020-03-26"),as.Date("2020-10-27"),1),cumdeath=cumdeath_pred_test,pred="Predicted")
## adjust by adding tmp_diff back
cumdeath_pred = cumdeath_pred %>% mutate(cumdeath = ifelse(date>=as.Date('2020-06-25'), cumdeath+tmp_dif, cumdeath))


cumdeath_dat=rbind(cumdeath_dat,cumdeath_pred)

cumdeath_test_pred=cumdeath_pred%>%filter(date >lastobsday)%>%filter(date <=lastobsday)

cumdeath_dat$pred = factor(cumdeath_dat$pred, levels = c('Train','Test', 'Predicted'))

#################################
###     Daily inc death      #####
#################################


incdeath_train=data.frame(date=pred_train$date,incdeath=pred_train$incdeath,pred="Train")
incdeath_train=incdeath_train%>%filter(date >=as.Date("2020-3-29"))%>%filter(date <=as.Date("2020-7-25"))


incdeath_dat=data.frame(date=pred_train$date,incdeath=pred_train$incdeath,pred="Train")
incdeath_dat=incdeath_dat%>%filter(date >=as.Date("2020-3-29"))


incdeath_pred=data.frame(date=seq(as.Date("2020-03-26"),as.Date("2020-10-27"),1),incdeath=incdeath_pred_test,pred="Predicted")
incdeath_pred=incdeath_pred%>%filter(date >=as.Date("2020-3-29"))%>%filter(date <=as.Date("2020-9-26"))



#################################
###     Weekly inc death      #####
#################################

target_date=seq(as.Date("2020-4-4"),as.Date("2020-09-26"),by=7)
incdeath_train_week=data.frame(date=target_date[1:17],weekdeath=colSums(matrix(incdeath_train$incdeath, nrow=7)),pred="Train")
# 
weekdeath_pred=data.frame(date=target_date,weekdeath=colSums(matrix(incdeath_pred$incdeath, nrow=7)),pred="Predicted")
weekdeath_dat=rbind(incdeath_train_week,weekdeath_pred)

## training error
sum(abs(incdeath_train$incdeath-incdeath_train_pred$incdeath))/nrow(incdeath_train)  ## 


# ## test error
# sum(abs(incdeath_test_pred$incdeath-incdeath_test_obs$incdeath))/nrow(incdeath_test_pred)  ## 

incdeath_dat$pred = factor(incdeath_dat$pred, levels = c('Train','Test', 'Predicted'))


#########################################################
##########          Compute     CI              #########
#########################################################

dat=read.csv("y_perm.csv")  ## permuation data
dat=t(dat)

dat=data.frame(date=seq(as.Date("2020-2-21"),as.Date("2020-2-21")+249,1),dat)

dat$date = as.Date(dat$date)

firstday = dat$date[1]

###  permuation indices of daily case and death
indices=read.csv("indices_mat2.csv")
indices=indices+1-34


lastobsday=as.Date('2020-7-31')


date=seq(as.Date("2020-03-26"),lastobsday,by=1)

permute_death=train_predicted[,c("date","cumdeath_pred_test","death_res","log_cumdeath_pred_test")]

death_perm=matrix(0,ncol(indices),1000)
for (perm in 1:1000){
  for (i in 1:ncol(indices)){
    death_perm[i,perm]=exp(permute_death$log_cumdeath_pred_test[i])+permute_death$death_res[indices[perm,i]]
  }
}

dat=dat[,-1]
perm_cumdeath_preds=NULL
perm_weekdeath_preds=NULL
perm_incdeath_preds=NULL
for (perm in 1:1000){
  cat(perm)
  cumcase=cumsum(dat[,perm])
  cumcase_lag21=lag(cumcase,21)
  cumcase_lag14=lag(cumcase,14)
  cumcase_lag7=lag(cumcase,7)
  cumcase_lag1=lag(cumcase,1)
  
  log_cumcase_pred_lag21=log(cumcase_lag21)
  log_cumcase_pred_lag14=log(cumcase_lag14)
  log_cumcase_pred_lag7=log(cumcase_lag7)
  log_cumcase_pred_lag1=log(cumcase_lag1)
  
  perm_data=data.frame(date=seq(as.Date("2020-2-21"),as.Date("2020-2-21")+249,1),log_cumcase_pred_lag1,log_cumcase_pred_lag7,log_cumcase_pred_lag14,log_cumcase_pred_lag21)
  
  perm_data=na.omit(perm_data)
  perm_data=perm_data%>%filter(date >="2020-03-26")
  
  perm_train=data.frame(perm_data%>%filter(date <=lastobsday),logcumdeath=log(death_perm[,perm]))
  
  reopen=ifelse(perm_train$date>="2020-5-1",1,0)
  
  perm_train=cbind(perm_train,reopen)
  
  reopen2=ifelse(perm_train$date>="2020-5-22",1,0)
  perm_train=cbind(perm_train,reopen2)
 
  reopen3=ifelse(perm_train$date>="2020-6-26",1,0)
  perm_train=cbind(perm_train,reopen3)
  
  
  lmfit=lm(logcumdeath~log_cumcase_pred_lag1+log_cumcase_pred_lag7+log_cumcase_pred_lag14+reopen+reopen:log_cumcase_pred_lag1+reopen:log_cumcase_pred_lag7+reopen:log_cumcase_pred_lag14+reopen2+reopen2:log_cumcase_pred_lag1+reopen2:log_cumcase_pred_lag7+reopen2:log_cumcase_pred_lag14+reopen3+reopen3:log_cumcase_pred_lag1+reopen3:log_cumcase_pred_lag7+reopen3:log_cumcase_pred_lag14,data=perm_train)
  
  
  ### do prediction
  perm_pred=perm_data%>%filter(date >=lastobsday+1)   ### need one more day for inc death (start from 8/1)
  reopen=ifelse(perm_pred$date>="2020-05-1",1,0)
  perm_pred=cbind(perm_pred,reopen)
  reopen2=ifelse(perm_pred$date>="2020-05-22",1,0)
  perm_pred=cbind(perm_pred,reopen2)
 
  reopen3=ifelse(perm_pred$date>="2020-06-26",1,0)
  perm_pred=cbind(perm_pred,reopen3)
  
  logcumdeath_pred=predict(lmfit, newdata = perm_pred)
  cumdeath_pred=exp(logcumdeath_pred)
  perm_cumdeath_preds=cbind(perm_cumdeath_preds,cumdeath_pred)
  
  
  
  ### calculate daily inc death
  incdeath_pred=diff(cumdeath_pred)  ## start from 8/2
  perm_incdeath_preds=cbind(perm_incdeath_preds,incdeath_pred)
  
  
  incdeath_pred=matrix(incdeath_pred,length(incdeath_pred),1)
  
  ### calculate weekly inc death
  incdeath_week=colSums(matrix(incdeath_pred, nrow=7))  ### last element do not use
  
  perm_weekdeath_preds=cbind(perm_weekdeath_preds,incdeath_week)
  
}


##############################
#########   plot           ###
##############################



#######################################
###### Cumulative death plot    #######
#######################################

label=cumdeath_dat$pred

cumdeath_dat=cbind(cumdeath_dat,label)

lower = apply(perm_cumdeath_preds[,],1,function(x){quantile(x,probs=0.025)})
upper = apply(perm_cumdeath_preds,1,function(x){quantile(x,probs=0.975)})

lower_dat = data.frame(date = lastobsday + c(0:(length(lower)-1)), 
                       cumdeath = lower+tmp_dif)
upper_dat = data.frame(date = lastobsday + c(0:(length(lower)-1)), 
                       cumdeath = upper+tmp_dif) 



knots = as.Date(c('2020-05-01','2020-5-22','2020-6-26'))

breakvec <-c(seq(from = as.Date("2020-03-26"),to=as.Date("2020-09-26"),by="20 days"))
p1 = ggplot() + 
  theme_pubr() +
  theme(legend.title = element_blank(), axis.text= element_text(size=15), axis.title = element_text(size=18, face="bold"), legend.text=element_text(size=15))+
  labs(y="Cumulative Death", x = "Date, 2020")+
  theme(panel.grid.major = element_line(size = 0.5, linetype = 3, colour = 'grey')) +
  geom_line(data = cumdeath_dat[cumdeath_dat$label!='Predicted',], aes(y = cumdeath, x = date, color=label, group=1), lwd = 1.5)+
  geom_line(data = cumdeath_dat[cumdeath_dat$label=='Predicted',], aes(y = cumdeath, x = date, color=label, group=1), lwd = 1.5)+
  geom_point(data = cumdeath_dat[cumdeath_dat$pred!='Predicted',], aes(y = cumdeath, x = date, color = pred, group = pred), size= 2.5) +
  geom_ribbon(data = lower_dat, aes(y = cumdeath, x = date, ymin=lower_dat$cumdeath, ymax=upper_dat$cumdeath),fill = 'red', alpha = 0.15) +
  geom_vline(xintercept =knots[1], linetype = 2, lwd =1, colour = 'grey40')+
  geom_vline(xintercept =knots[2], linetype = 2, lwd =1, colour = 'grey40')+
  geom_vline(xintercept =knots[3], linetype = 2, lwd =1, colour = 'grey40')+
  coord_cartesian(ylim=c(0, 200000)) + 
  scale_color_manual(breaks=c("Train","Test","Predicted"),labels = c("Training data \nused in model fitting","Testing data \nnot included in model fitting","Predicted"),
                     values = c(
                       'Predicted' = '#999999',
                       'Train' = 'deepskyblue3',
                       'Test' = 'lightcoral'
                     )) +
  scale_x_date(expand=c(0,0),breaks = breakvec,
               labels=date_format("%b-%d"),
               limits = as.Date(c('2020-03-26','2020-09-26')))

p1


png("Figure/US_cumdeath_071920.png",width=750,height=480)
p1
dev.off()



#######################################
###### Daily inc death plot    #######
#######################################

label=incdeath_dat$pred

incdeath_dat=cbind(incdeath_dat,label)

lower = apply(perm_incdeath_preds,1,function(x){quantile(x,probs=0.025)})
upper = apply(perm_incdeath_preds,1,function(x){quantile(x,probs=0.975)})



lower_dat = data.frame(date =lastobsday+1 + c(0:(length(lower)-1)), 
                       incdeath = lower)
upper_dat = data.frame(date = lastobsday+1 + c(0:(length(lower)-1)), 
                       incdeath = upper) 

breakvec <-c(seq(from = as.Date("2020-03-26"),to=as.Date("2020-09-15"),by="20 days"))
p1 = ggplot() + 
  theme_pubr() +
  theme(legend.title = element_blank(), axis.text= element_text(size=15), axis.title = element_text(size=18, face="bold"), legend.text=element_text(size=15))+
  labs(y="Daily Inc Death", x = "Date, 2020")+
  theme(panel.grid.major = element_line(size = 0.5, linetype = 3, colour = 'grey')) +
  # geom_line(data = dat, aes(y = new_obs, x = date, colour = pred, group = pred), lwd = 1.5) +
  geom_line(data = incdeath_dat[incdeath_dat$label!='Predicted',], aes(y = incdeath, x = date, color=label, group=1), lwd = 1.5)+
  geom_line(data = incdeath_dat[incdeath_dat$label=='Predicted',], aes(y = incdeath, x = date, color=label, group=1), lwd = 1.5)+
  geom_point(data = incdeath_dat[incdeath_dat$pred!='Predicted',], aes(y = incdeath, x = date, color = pred, group = pred), size= 2.5) +
  # geom_ribbon(data = lower_dat, aes(y = incdeath, x = date, ymin=lower_dat$incdeath, ymax=upper_dat$incdeath),fill = 'red', alpha = 0.15) +
  geom_vline(xintercept =knots[1], linetype = 2, lwd =1, colour = 'grey40')+
  geom_vline(xintercept =knots[2], linetype = 2, lwd =1, colour = 'grey40')+
  geom_vline(xintercept =knots[3], linetype = 2, lwd =1, colour = 'grey40')+
  coord_cartesian(ylim=c(0, 3000)) + 
  scale_color_manual(breaks=c("Train","Test","Predicted"),labels = c("Training data \nused in model fitting","Testing data \nnot included in model fitting","Predicted"),
                     values = c(
                       'Predicted' = '#999999',
                       'Train' = 'deepskyblue3',
                       'Test' = 'lightcoral'
                     )) +
  scale_x_date(expand=c(0,0),breaks = breakvec,
               labels=date_format("%b-%d"),
               limits = as.Date(c('2020-03-26','2020-09-15')))

p1



png("Figure/US_daily_incdeath_071920.png",width=750,height=480)
p1
dev.off()



#######################################
###### Weekly inc death plot    #######
#######################################

### plot

label=weekdeath_dat$pred

weekdeath_dat=cbind(weekdeath_dat,label)

lower = apply(perm_weekdeath_preds[1:13,],1,function(x){quantile(x,probs=0.025)})
upper = apply(perm_weekdeath_preds[1:13,],1,function(x){quantile(x,probs=0.975)})


# target_date=seq(as.Date("2020-07-18"),as.Date("2020-07-18")+14*7,by=7)

target_date=as.Date(c("2020-08-08",
                      "2020-08-15","2020-08-22","2020-08-29","2020-09-05","2020-09-12","2020-09-19","2020-09-26"))

lower_dat = data.frame(date =target_date, 
                       weekdeath = lower)
upper_dat = data.frame(date = target_date, 
                       weekdeath = upper) 


                        


knots = as.Date(c('2020-05-01','2020-5-22','2020-6-26'))

breakvec <-c(seq(from = as.Date("2020-03-26"),to=as.Date("2020-09-15"),by="20 days"))
p1 = ggplot() + 
  theme_pubr() +
  theme(legend.title = element_blank(), axis.text= element_text(size=15), axis.title = element_text(size=18, face="bold"), legend.text=element_text(size=15))+
  labs(y="Weekly Inc Death", x = "Date, 2020")+
  theme(panel.grid.major = element_line(size = 0.5, linetype = 3, colour = 'grey')) +
  geom_line(data = weekdeath_dat[weekdeath_dat$label!='Predicted',], aes(y = weekdeath, x = date, color=label, group=1), lwd = 1.5)+
  geom_line(data = weekdeath_dat[weekdeath_dat$label=='Predicted',], aes(y = weekdeath, x = date, color=label, group=1), lwd = 1.5)+
  geom_point(data = weekdeath_dat[weekdeath_dat$pred!='Predicted',], aes(y = weekdeath, x = date, color = pred, group = pred), size= 2.5) +
  geom_point(data = weekdeath_dat[weekdeath_dat$pred=='Predicted',], aes(y = weekdeath, x = date, color = pred, group = pred), size= 2.5) +
  geom_ribbon(data = lower_dat, aes(y = weekdeath, x = date, ymin=lower_dat$weekdeath, ymax=upper_dat$weekdeath),fill = 'red', alpha = 0.15) +
  geom_vline(xintercept =knots[1], linetype = 2, lwd =1, colour = 'grey40')+
  geom_vline(xintercept =knots[2], linetype = 2, lwd =1, colour = 'grey40')+
  geom_vline(xintercept =knots[3], linetype = 2, lwd =1, colour = 'grey40')+
  coord_cartesian(ylim=c(0, 16000)) + 
  scale_color_manual(breaks=c("Train","Test","Predicted"),labels = c("Training data \nused in model fitting","Testing data \nnot included in model fitting","Predicted"),
                     values = c(
                       'Predicted' = '#999999',
                       'Train' = 'deepskyblue3',
                       'Test' = 'lightcoral'
                     )) +
  scale_x_date(expand=c(0,0),breaks = breakvec,
               labels=date_format("%b-%d"),
               limits = as.Date(c('2020-03-26','2020-09-15')))

p1


png("Figure/US_weekly_incdeath_071920.png",width=750,height=480)
p1
dev.off()




