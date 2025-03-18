-- MySQL dump 10.13  Distrib 8.0.33, for Win64 (x86_64)
--
-- Host: localhost    Database: fitmyvibe
-- ------------------------------------------------------
-- Server version	8.0.33

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `articles`
--

DROP TABLE IF EXISTS `articles`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `articles` (
  `id` int NOT NULL,
  `budget` int NOT NULL,
  `gender` varchar(10) NOT NULL,
  `masterCategory` varchar(20) NOT NULL,
  `subCategory` varchar(20) NOT NULL,
  `articleType` varchar(20) NOT NULL,
  `baseColor` varchar(20) NOT NULL,
  `season` varchar(10) NOT NULL,
  `fashionYear` int NOT NULL,
  `usage` varchar(10) NOT NULL,
  `productDisplayName` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `articles`
--

LOCK TABLES `articles` WRITE;
/*!40000 ALTER TABLE `articles` DISABLE KEYS */;
INSERT INTO `articles` VALUES (3954,20,'Women','Apparel','Topwear','Tshirts','Pink','Summer',2011,'Casual','Jealous 21 Women\'s Pink T-shirt'),(9204,100,'Men','Footwear','Shoes','Casual Shoes','Black','Summer',2011,'Casual','Puma Men Future Cat Remix SF Black Casual Shoes'),(12369,30,'Men','Apparel','Topwear','Shirts','Purple','Fall',2011,'Formal','Reid & Taylor Men Check Purple Shirts'),(15970,50,'Men','Apparel','Topwear','Shirts','Navy Blue','Fall',2011,'Casual','Turtle Check Men Navy Blue Shirt'),(18005,30,'Men','Apparel','Bottomwear','Shorts','Black','Summer',2011,'Sports','Puma Men Long Logo Black Bermuda'),(23278,120,'Men','Accessories','Watches','Watches','Off White','Winter',2016,'Casual','Maxima Ssteele Men Off White Watch'),(28456,45,'Women','Apparel','Bottomwear','Skirts','Blue','Summer',2012,'Sports','Urban Yoga Women Blue Skirt With Leggings'),(39386,40,'Men','Apparel','Bottomwear','Jeans','Blue','Summer',2012,'Casual','Peter England Men Party Blue Jeans'),(43369,25,'Men','Footwear','Flip Flops','Flip Flops','Black','Summer',2013,'Casual','Reebok Men Black Possession Flip Flops'),(47957,60,'Women','Accessories','Bags','Handbags','Blue','Summer',2012,'Casual','Murcia Women Blue Handbag'),(54118,50,'Women','Footwear','Shoes','Heels','Black','Winter',2012,'Casual','Rocia Women Black Flats'),(59263,80,'Women','Accessories','Watches','Watches','Silver','Winter',2016,'Casual','Titan Women Silver Watch');
/*!40000 ALTER TABLE `articles` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-03-18  2:30:40
