
Inspiration
•	What can we learn about different hosts and areas?
•	What can we learn from predictions? (ex: locations, prices, reviews, etc)
•	Which hosts are the busiest and why?
•	Is there any noticeable difference of traffic among different areas and what could be the reason for it?

data.columns
Out[16]: 
Index(['id', 'name', 'host_id', 'host_name', 'neighbourhood_group',
       'neighbourhood', 'latitude', 'longitude', 'room_type', 'price',
       'minimum_nights', 'number_of_reviews', 'last_review',
       'reviews_per_month', 'calculated_host_listings_count',
       'availability_365'],
      dtype='object')
I’ve started studding phyton recently and tried applied what I’ve learned in a real data, therefore, I find one Airbnb data from Kaggle( https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) and do some exploratory analysis, data cleaning and regression prediction.

Before diving into the data, we can take a look at the inspiration questions and have a few hypotheses regarding to them. Afterwards, we can explore the data to prove our hypotheses.
•	What can we learn about different hosts and areas? 

About hosts: generally, I think there are two type of hosts: professional hosts, whose name appears repeatedly under name column, which means they might have several apartments/houses with different room types in different neighbourhood. Amateur hosts, who lease their vacant space at home.  Professional hosts are probably getting more reviews and post more listings than amateur hosts as they care more about their return in the business (we need occupation rate to prove it).

About the areas: Demand affects supply, therefore, famous area with more iconic spots has more Airbnb supply as most customers are travelers. Its average price will also be higher. On the other hand,  the area adjacent to the popular area should also has comparatively more Airbnb supply as it has a balance of distance and cost.  
Findings: a. Manhattan and Brooklyn account over 85% of all New York Airbnb.
In: data.neighbourhood_group.value_counts().plot(kind='bar',title='NYC Airbnb Counts')

 

In: data.neighbourhood_group.value_counts()/data.neighbourhood_group.notnull().sum()
Out[7]: 
Manhattan        0.443011
Brooklyn         0.411167
Queens           0.115881
Bronx            0.022313
Staten Island    0.007629

 b. the price range of these two areas are quite different as the average price/night of in Manhattan is 72UCD more expensive than in Brooklyn. 

 
The yellow is Brooklyn data while blue is Manhattan, we can see overall the frequency of Brooklyn are more to the left, which means it has lower average.


Is there any noticeable difference of traffic among different areas and what could be the reason for it?
As we can see from the scatter plot, both Manhattan and part of Brooklyn are the densest area. As I mentioned in the first part, besides the historical spots and cost efficiency. 
The price range are so large and there are a lot outliner. So I will reduce the outliner by getting rid of the upper and lower 0.5%. Meanwhile, we find it’s hard to see from scatter plot where are more expensive as the price range is too large. I also reduce to price under 200 and show graph as below.
  

After adjustment, we can get an idea of both the price difference and density of available Airbnb.
 
 


What can we learn from predictions? (ex: locations, prices, reviews, etc)

Before exploring the data, I think price is affected by area, host, room type, min nights, reviews. Therefore, we can create linear regression with the factors to predict price. Reviews show popularity-which could because of good location or good price, availability, so it’s not included in the regression. 

Which hosts are the busiest and why?
I think there are a few factors that make some hosts busier than others, that is, apartment/house numbers, minimum nights, calculated host listings count, and availability. Review numbers and last review could also be the reason as travelers tends to ask more questions if the place has fewer review or not review for a while. 
