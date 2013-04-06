/*
Lab 11
This lab is to learn to use structures and incorporate them into the things 
you've been doing. You will implement a Card structure and several functions 
that allow you to create cards, print cards, create a deck, shuffle the deck, 
and sort the deck.

TASKS
 1) create a card structure - suit and value
 2) create a compareTo function
    -compares the value of two cards a and b
    -returns the -1 if a < b, 1 if a>b, and 0 if a==b
 3) create a printCard function
    -takes a single card as input and prints the card in a readable format
 4) Function to initialize a deck of cards
    -should fill an array of 52 cards with the 52 unique cards
    -use your createCard function
 5) create a function to print the whole deck of cards
    -use your printCard function
 6) create a function to shuffle the deck 
    -use the rand and srand(time(NULL)) functions to make the shuffle random.
    -use the print statements to make sure the 
 7) create a function to sort the deck
    -use print statements to ensure the cards are sorted
    
NOTE
    - your main function should use only high-level function calls.
    - main must create the deck, initialize it, shuffle it and sort it.
 
 @uthor: Shane Griffith (CprE 185. Lab 11. Fall 2008.)
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//define the suits and face value cards
#define HEARTS 3
#define SPADES 2
#define DIAMONDS 1
#define CLUBS 0

#define ACE 14
#define KING 13
#define QUEEN 12
#define JACK 11

//define the size of the deck
#define DECKSIZE 52

//create a card structure
//notice the use of 'typedef'
typedef struct
{
      int suit;
      int value;
}card;

/* given a suit and a value, this function returns a card structure.
 *
 */
card createCard(int suit, int value)
{
     card x;
     x.suit = suit;
     x.value = value;
     return x;
}

/*CompareTo compares two cards based on their values.
 *
 */
int compareTo(card a, card b)
{
    //The card's value is more significant than the suit, so compare value first.
    if(a.value > b.value)
    {
         return 1;
    }
    else if(a.value < b.value)
    {
         return -1;
    }
    else if(a.suit > b.suit)
    {
         return 1;
    }
    else if(a.suit < b.suit)
    {
         return -1;
    }
    return 0;
}

/* Print the card according to the value and the suit.
 *  given 5 of hearts, a 5_H is printed
 *  given 11 (Jack) of diamonds, a J_D is printed
 */
void printCard(card a)
{
     //print the card
     switch(a.value)
     {
          case ACE: printf("A_"); break;
          case KING: printf("K_"); break;
          case QUEEN: printf("Q_"); break;
          case JACK: printf("J_"); break;  
          default: printf("%d_", a.value);
     }
     
     switch(a.suit)
     {
          case HEARTS: printf("H "); break;
          case SPADES: printf("S "); break;
          case DIAMONDS: printf("D "); break;
          case CLUBS: printf("C "); break;
     }
     //puts("");
}

/* The input is an array of cards, which are not initialized. The output is 52
 * cards that are initialized to some values
 */
void initializeDeck(card deck[])
{
     int i;
     card temp;
     
     for(i=0; i<DECKSIZE; i++)
          deck[i] = createCard(i/13, i%13+2);
}

/* Print an array of cards.
 *
 */
void printDeck(card deck[])
{
     int i;
     for(i=0; i< DECKSIZE; i++)
          printCard(deck[i]);
     printf("\n\n");
}

/* Swap two cards in the deck. Used to shuffle and to sort the deck.
 *
 */
void swap(card deck[], int idx1, int idx2)
{
     card temp = deck[idx1];
     deck[idx1] = deck[idx2];
     deck[idx2] = temp;
}

/* Shuffle the cards in the deck.
 *
 */
void shuffleDeck(card deck[])
{
     int i;
     srand(time(NULL));
     for(i=0; i<DECKSIZE; i++)
         swap(deck, rand()%(DECKSIZE-i), DECKSIZE-i-1);
}


//find the index of the minimum card between idx1 incl and idx2 excl
//for sorting...

/* Find the value of the minimum card between the given two indices. This is
 * useful for sorting the cards with selection sort.
 */
int findMinCard(card deck[], int idx1, int idx2)
{
     int i;
     int minidx=idx1;
     
     for(i=idx1; i<idx2; i++)
          if(compareTo(deck[minidx], deck[i]) == -1)
               minidx = i;
     
     return minidx; //maybe return idx here...  
}

/* Sort the deck using selection sort.
 *
 */
void  sortDeck(card deck[])
{
      int i, minidx;
      for(i=0; i<DECKSIZE; i++)
      {
          //find the index of the minimum card between i and the end of the deck
          minidx = findMinCard(deck, i, DECKSIZE);
          
          //place the minimum card at the ith position in the deck
          swap(deck, i, minidx);
      }
}



int main()
{
    //create the deck
    card deck[DECKSIZE];
    
    //initialize the deck
    initializeDeck(deck);
    printDeck(deck);
    
    //shuffle the deck
    shuffleDeck(deck);
    printDeck(deck);
    
    //sort the deck
    sortDeck(deck);
    printDeck(deck);
    
    printf("\n");
    system("pause");
}




