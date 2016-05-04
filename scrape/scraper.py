from bs4 import BeautifulSoup
import requests
import pandas as pd
import urllib
import numpy as np
import datetime
import time

class EtsyScraper(object):

    """
    Custom made scraper of Etsy using BeautifulSoup. Given links to Etsy stores, scrapes
    images and related imformation and downloads images with encoded filenames.
    """

    def __init__(self, ii_dict):
        '''
        INPUT: Dict
        OUTPUT: None

        Takes in a dictionart (ii_dict) that contains Etsy store names as keys,
        image index location (int) and whether the titles contain tags that have
        and apostrophe (e.g. "90's" instead of "90s"). This dict in created in
        load_data.
        '''
        self.list_ids = set()
        self.results = []
        self.df = pd.DataFrame(columns=['item_id', 'shop', 'name', 'label', 'url', 'price'])
        self.url = 'https://www.etsy.com/listing/'
        self.ii_dict = ii_dict

    def get_listings(self, link):
        '''
        INPUT: String
        OUTPUT: None

        Iterates through all Etsy stores and returns suitable unique Item_IDs
        '''
        x = requests.get(link)
        soup = BeautifulSoup(x.content, "lxml")
        listid_tag = soup.find_all(class_="block-grid-item listing-card position-relative")
        try:
            store = str(soup.find_all(class_="mb-xs-1")[0])[32:-5]
        except:
            store = str(soup.find_all(class_="hide-xs hide-sm")[0]).split('\n')[1].strip()
        for item in range(len(listid_tag)):
            self.list_ids.add((listid_tag[item]['data-palette-listing-id'], store))
        print "%d total listenings found" % len(self.list_ids)

    def scrape_all_listings(self):
        '''
        INPUT: None
        OUTPUT: None

        Iterates through list of listings and retreives photos, tags, and
        other relavent item information.
        '''
        count = 0
        for listing in self.list_ids:
            self.scrape_one_listing(listing[0], self.ii_dict[listing[1]][0], count,
                                    apost=self.ii_dict[listing[1]][1])
            count += 1
            percent = float(count)/len(self.list_ids)*100
            print "Scraped %d of %d listings: %f percent complete. Current shop is: %s"
                   % (count, len(self.list_ids), percent, listing[1])
        print "Scraping of listings has completed."

    def scrape_one_listing(self, listing, image_index, count, apost=False):
        '''
        INPUT: String, Int, Int, Boolean
        OUTPUT:

        Called by scrape_all_listings. Scrapes specific fields using BeautifulSoup
        '''
        x = requests.get(self.url+str(listing))
        soup = BeautifulSoup(x.content, "lxml")
        try:
            store = str(soup.find_all(itemprop="title")[0])[23:-7]
        except:
            store = ""
        try:
            title = str(soup.find_all(itemprop="name")[0])[22:-7]
        except:
            title = ""
        try:
            price = np.float(str(soup.find_all(class_="currency-value")[0])[29:-7])
        except:
            price = ""
        image_input = "image-" + str(image_index)
        try:
            image = soup.find_all(id=image_input)[0]['data-full-image-href']
        except:
            try:
                image = soup.find_all(id="image-0")[0]['data-full-image-href']
            except:
                image = ""
                print "Warning: Failed to find desired image from %s at index %i" % (store, image_index)
        if apost == False:
            tag = self.item_tagger_v1(title)
        else:
            tag = self.item_tagger_v2(title)
        if tag:
            self.results.append((listing, store, title, tag, image, price))

    def store_in_dataframe(self):
        '''
        INPUT: None
        OUTPUT: None

        Imports results into Pandas DataFrame.
        '''
        print "Storing scraped data in a Pandas DataFrame..."
        self.df = self.df.append(pd.DataFrame.from_records(self.results, columns=['item_id', 'shop', 'name', 'label', 'url', 'price'])).reset_index()
        self.df = self.df[['item_id', 'shop', 'name', 'label', 'url', 'price']]
        self.df = self.df.drop_duplicates()
        print "DataFrame creation completed."

    def pickle_dataframe(self):
        '''
        INPUT: None
        OUTPUT: None

        Writes Pandas DataFrame of results to pickle file.
        '''
        print "Storing Pandas DataFrame in pickle file..."
        dt = "--".join(str(datetime.datetime.now())[:-7].split())
        self.df.to_pickle('/home/ubuntu/project/scrape/meta_data-{}.pkl'.format(dt))
        print "DataFrame pickling completed."

    def download_links(self, row):
        '''
        INPUT: DataFrame
        OUTPUT: None

        Helper function that used to download images in download_images.
        '''
        item_id = str(row.item_id)
        label = str(row.label)
        url = str(row.url)
        try:
            urllib.urlretrieve(url,
                '/home/ubuntu/project/scrape/images/{}_{}.jpg'.format(label, item_id))
        except:
            pass

    def download_images(self):
        '''
        INPUT: None
        OUTPUT: None

        Downloads images.
        '''
        print "Downloading scraped images..."
        self.df.apply(lambda x: self.download_links(x), axis=1)
        print "Downloading of images completed."

    def item_tagger_v1(self, title):
        '''
        INPUT: String
        OUTPUT: String

        Returns tag from title if only one tag is present in the title.
        Titles without apostrphe (e.g. "90s not 90's")
        '''
        classes = ['50s', '60s', '70s', '80s', '90s']
        tags = []
        for elem in classes:
            if elem in title:
                tags.append(elem)
        if len(tags) == 1:
            return tags[0]

    def item_tagger_v2(self, title):
        '''
        INPUT: String
        OUTPUT: String

        Returns tag from title if only one tag is present in the title.
        Titles without apostrphe (e.g. "90's not 90s")
        '''
        classes = ["50's", "60's", "70's", "80's", "90's"]
        tags = []
        for elem in classes:
            if elem in title:
                tags.append(elem)
        if len(tags) == 1:
            return tags[0]

def load_data():
    '''
    INPUT: None
    OUTPUT: None

    Loads data.
    '''
    df = pd.read_pickle('/home/ubuntu/project/scrape/etsy_data.pkl')
    ii_dict = {}
    df_cleaned = df[pd.notnull(df['Image_Index'])]
    for i in df_cleaned.iterrows():
        ii_dict[i[1]['Shop']] = [int(i[1]['Image_Index']), i[1]['Apost']]
    listing_links = list(df.Link)
    return (ii_dict, listing_links)

if __name__ == "__main__":

    # Load in data of pre-compiled of Etsy Stores to scrape
    ii_dict, listing_links = load_data()

    # Instanciate EtsyScraper
    scrape = EtsyScraper(ii_dict)

    # Get listings stored in ii_dict
    for link in listing_links:
        scrape.get_listings(link)
        time.sleep(3)

    # Make sure we got them all!
    for link in listing_links:
        scrape.get_listings(link)
        time.sleep(3)

    # For each listing, scrape listing information and image links
    scrape.scrape_all_listings()

    # Store the results from scraping in a Pandas DataFrame
    scrape.store_in_dataframe()

    # Download images info
    scrape.download_images()

    # Pickle Pandas DataFrame
    scrape.pickle_dataframe()

    print "Scraping session completed successfully."
