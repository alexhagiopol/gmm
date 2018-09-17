pkg load statistics
clear
clc

image = imread('party_spock.png');
[imRows, imCols] = size(image); 
image = im2double(image);   %convert image from 0-255 int array to 0-1 decimal array
components = 3;             %Change this to set number of segments
iterations = 10;
means = zeros(1,components); %initialize means with random pixel values
for i = 1:components
    randCol = 1 + round((imCols-1)*rand(1,1));
    randRow = 1 + round((imRows-1)*rand(1,1));
    means(i) = image(randRow, randCol);
end
variances = ones(1,components)/255;           %Initial variance you asked for is 1. Because of my conversion, made it 1/255
stdevs = sqrt(variances);
weights = ones(1,components);
numPoints = zeros(1,components);


%initialize matrices equal to image size
jointProb = zeros(imRows, imCols);
meanValues = zeros(imRows, imCols);
probDensity = zeros(imRows, imCols, components);

for x = 1:iterations
    %E Step from page 818 of AI book
    componentCorrelationMatrix = zeros(imRows, imCols); %counts number of points associated with ea component
    prevHighestProb = zeros(imRows, imCols);
    for i = 1:components        
        probDensity(:,:,i) = weights(i)*pdf('norm', image, means(i), stdevs(i));  %matlab stat toolbox function that computes gaussian
        probDensity(:,:,i) = probDensity(:,:,i) / max(max(probDensity(:,:,i)));
        probDensity(:,:,i) = floor(probDensity(:,:,i)*1000)/1000;
        testDensity = probDensity(:,:,i);
        test = log(testDensity);
        %jointProb = jointProb + probDensity(:,:,i);                %computes sum in eqn (1)
        for j = 1:imRows
            for k = 1:imCols
                if probDensity(j,k,i) >= prevHighestProb(j,k)  %finds & stores maximum a posteriori probability
                     prevHighestProb(j,k) = probDensity(j,k,i);
                     meanValues(j,k) = means(i);            %assigns mean value of max a post prob
                     componentCorrelationMatrix(j,k) = i;   %stores which point "belongs" to which segment
                end
            end
        end       
    end
     
    %count the number of points assigned to each segment
    for j = 1:imRows
        for k = 1:imCols
            numPoints(componentCorrelationMatrix(j,k)) = numPoints(componentCorrelationMatrix(j,k)) + 1;
        end
    end
    %M Step from page 819 of AI book
    %compute new means
    for i = 1:components
        means(i) = sum(sum(probDensity(:,:,i).*image/numPoints(i))); 
        variances(i) = sum(sum((probDensity(:,:,i).*(image - means(i)).^2)/numPoints(i)));
        weights(i) = numPoints(i)/(imRows*imCols);
    end
    stdevs = sqrt(variances);
    
    
end
imshow(meanValues, [min(meanValues(:)), max(meanValues(:))]);




