close all
% data number m
% real theta0=1, theta1=0.5
m=11;
x=linspace(0,10,m);
e=(rand(1,m)-0.5)*2;
y=1+0.5*x+e;
%%
sample=1000;
theta0=linspace(0,2,sample);
theta1=linspace(0,1,sample);
%theta0=repmat(theta0,m,1);
%theta1=repmat(theta1,m,1);
h=zeros(sample,sample);
%%
for i=1:sample
    for j=1:sample
        h(i,j)=1/(2*m)*sum((theta0(i)+x*theta1(j)-y).^2);
    end
end
%%
[h_min,I]=min(h(:));
[theta0_op,theta1_op]=ind2sub(size(h),I);
theta0_op=theta0(theta0_op)
theta1_op=theta1(theta1_op)
%%
plot(x,y,'ro')
hold on
y_pred=theta0_op+theta1_op*x;
plot(x,y_pred,'b')
legend('data set','prediction')
%%
figure
[X Y]=meshgrid(theta0,theta1);
contour(X,Y,log(h'))
hold on
plot(theta0_op,theta1_op,'rx')
xlabel('\theta_0');
ylabel('\theta_1');
%%
% syms theta0 theta1
% ezsurf(theta0,theta1,h)
% xlabel('\theta_0');
% ylabel('\theta_1');
% zlabel('cost function');
% title('cost function of \theta_0 and \theta_1')