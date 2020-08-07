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

### ### ### ### ### ### ### ### ### 
###  Example of extracting JHU data 
### ### ### ### ### ### ### ### ### 


library (readr)
urlfile="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
jhu_case<-read_csv(url(urlfile))
colnames(jhu_case)
CA = jhu_case %>% filter(Province_State=='California')  ### extract CA case

CA_cumcase = CA[,c(12:length(CA))] %>% apply(2, sum)   ### California cumlative death
CA_case = diff(CA_cumcase)  ## California daily death

### US cumulative case
us_case = jhu_case[,c(12:length(jhu_case))] %>% apply(2, sum)  ### add the case number of each date

## US daily case
us_case=diff(us_case)

### extract US daily case numbers
case=as.numeric(us_case)
casedate=names(us_case)





### ### ### ### ### ### ### ### ### 
### ###  Plot Daily Case  ### ### 
### ### ### ### ### ### ### ### ### 

### read our data including training data and predicted daily case for plotting
dat = read.csv("US_death9/all_obs_param30.csv")  ## our data file (format: first column: date; second column: new_obs observed daily case, third column: pred (label indicates Train or Predicted or Test))

dat$date = as.Date(dat$date)

firstday = dat$date[1]
pred=dat[dat$pred=="Predicted",c("date","new_obs","pred")]

lastobsday=as.Date("2020-07-31")

dat$pred = factor(dat$pred, levels = c('Train', 'Test', 'Predicted'))

label=dat$pred

dat=cbind(dat,label)


knots = as.Date(c('2020-3-13','2020-3-27','2020-4-10','2020-04-24','2020-05-01','2020-5-22','2020-6-26'))


r0 = as.matrix(read.csv("US_death9/y_perm.csv"))
lower = apply(r0, 2, function(x) quantile(x, probs=0.025))
upper = apply(r0, 2, function(x) quantile(x, probs=0.975))

lower_dat = data.frame(date = as.Date(firstday, '%m/%d/%y') + c(0:(length(lower)-1)), 
                       new_obs = lower) %>% filter(date >= lastobsday)
upper_dat = data.frame(date = as.Date(firstday, '%m/%d/%y') + c(0:(length(lower)-1)), 
                       new_obs = upper) %>% filter(date >= lastobsday)

breakvec <-c(seq(from = as.Date("2020-02-21"),to=as.Date("2020-10-3"),by="20 days"))
p1 = ggplot() + 
  theme_pubr() +
  theme(legend.title = element_blank(), axis.text= element_text(size=15), axis.title = element_text(size=18, face="bold"), legend.text=element_text(size=15))+
  labs(y="Daily New Cases", x = "Date, 2020")+
  theme(panel.grid.major = element_line(size = 0.5, linetype = 3, colour = 'grey')) +
  # geom_line(data = dat, aes(y = new_obs, x = date, colour = pred, group = pred), lwd = 1.5) +
  geom_line(data = dat[dat$label!='Predicted',], aes(y = new_obs, x = date, color=label, group=1), lwd = 1.5)+
  geom_line(data = dat[dat$label=='Predicted',], aes(y = new_obs, x = date, color=label, group=1), lwd = 1.5)+
  geom_point(data = dat[dat$pred!='Predicted',], aes(y = new_obs, x = date, color = pred, group = pred), size= 2.5) +
  # guides(colour = guide_legend(override.aes = list(shape = c(16, 16, NA)))) + 
  geom_ribbon(data = lower_dat, aes(y = new_obs, x = date, ymin=lower_dat$new_obs, ymax=upper_dat$new_obs),fill = 'red', alpha = 0.15) + 
  geom_vline(xintercept =knots[1], linetype = 2, lwd =1, colour = 'grey40')+
  geom_vline(xintercept =knots[2], linetype = 2, lwd =1, colour = 'grey40')+
  geom_vline(xintercept =knots[3], linetype = 2, lwd =1, colour = 'grey40')+
  geom_vline(xintercept =knots[5], linetype = 2, lwd =1, colour = 'grey40')+
  geom_vline(xintercept =knots[6], linetype = 2, lwd =1, colour = 'grey40')+
  geom_vline(xintercept =knots[7], linetype = 2, lwd =1, colour = 'grey40')+
  coord_cartesian(ylim=c(0, 100000)) + 
  scale_color_manual(breaks=c("Train","Test","Predicted"),labels = c("Training data \nused in model fitting","Testing data \nnot included in model fitting","Predicted"),
                     values = c(
                       'Predicted' = '#999999',
                       'Train' = 'deepskyblue3',
                       #                       # 'Predicted' = 'deepskyblue3',
                       #                       # 'Train' = '#999999',
                       'Test' = 'lightcoral')) +
  # scale_x_discrete(limits=c("Train", "Test", "Predicted"))+
  # scale_fill_discrete(name = "", labels = c("Training data \nused in model fitting", "Testing data \nnot included in model fitting", "Predicted"))+
  scale_x_date(expand=c(0,0),breaks = breakvec,
               labels=date_format("%b-%d"),
               limits = as.Date(c('2020-02-21','2020-10-3')))


png("US_death9/US_daily_case_080220.png",width=750,height=480)
p1
dev.off()



