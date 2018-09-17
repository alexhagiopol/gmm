pkg load statistics
clear
clc

image = imread('party_spock.png');
[imRows, imCols] = size(image); 
image = im2double(image);   %convert image from 0-255 int array to 0-1 decimal array
components = 8;             %Set components
iterations = 30;           %Set iterations
means = zeros(1,components); %initialize means with random pixel values
for i = 1:components
    randCol = 1 + round((imCols-1)*rand(1,1));
    randRow = 1 + round((imRows-1)*rand(1,1));
    means(i) = image(randRow, randCol);
end
variances = ones(1,components)/255;           %Initial variance you asked for is 1. Because of my conversion, made it 1/255
stdevs = sqrt(variances);
weights = ones(1,components);

logLikelyhood = zeros(1,iterations);
for i = 1:iterations
    %E step
    responsibilities = zeros(imRows, imCols, components);
    denominator = zeros(imRows, imCols);             %denominator of responsibilities equation 9.13
    for k = 1:components
        denominator = denominator + weights(k)*pdf('norm', image, means(k), stdevs(k));
    end    
    for k = 1:components
        responsibilities(:,:,k) = weights(k)*pdf('norm', image, means(k), stdevs(k))./denominator; %compute responsibilities eqn 9.13
    end
       
    %M step
    numPoints = zeros(1,components);    
    numPoints(:) = sum(sum(responsibilities(:,:,:))); %compute Nk
    
    for k = 1:components
        means(k) = sum(sum(responsibilities(:,:,k).*image))/numPoints(k); %update means
        variances(k) = sum(sum(responsibilities(:,:,k).*(image - means(k)).^2))/numPoints(k); %update variances
        weights(k) = numPoints(k)/(imRows*imCols); %update weights
    end    
    logLikelyhood(i) = sum(sum(log(denominator))); %compute eqn 9.14 and store   
end

%display results
displayMatrix = zeros(imRows, imCols);
for i = 1:imRows
    for j = 1:imCols
        prevMaxResp = 0;
        for k = 1:components
            if responsibilities(i,j,k) >= prevMaxResp
                displayMatrix(i,j) = means(k);
                prevMaxResp = responsibilities(i,j,k);
            end
        end
    end
end
imshow(displayMatrix, [min(displayMatrix(:)), max(displayMatrix(:))]);
figure;
plot(1:iterations,logLikelyhood,'r+');














